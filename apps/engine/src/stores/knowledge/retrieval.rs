//! Hybrid retrieval over the chunks table: pgvector dense and full-text lexical, fused with RRF.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

use crate::core::traits::knowledge::retrieval::{Embedder, Retriever};
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::knowledge::retrieval::{RetrievalError, RetrievalQuery};
use crate::stores::postgres::{quote_literal, vector_literal};

/// How the two retrieval legs are run and fused. Built from the retrieval section; there is no Default.
#[derive(Debug, Clone)]
pub struct RetrievalTuning {
    /// Candidates pulled from each leg before fusion.
    pub candidates: i64,
    /// Reciprocal rank fusion constant.
    pub rrf_k: f32,
    /// PostgreSQL text search configuration for the lexical leg.
    pub text_search_config: String,
    /// Run the pgvector leg.
    pub dense: bool,
    /// Run the full-text leg.
    pub lexical: bool,
    /// Drop fused results below this score.
    pub min_score: f32,
    /// Drop a chunk when the summary covering it is already in the result.
    pub collapse_tree: bool,
}

impl From<&crate::core::config::Retrieval> for RetrievalTuning {
    fn from(cfg: &crate::core::config::Retrieval) -> Self {
        Self {
            candidates: cfg.candidates,
            rrf_k: cfg.rrf_k,
            text_search_config: cfg.text_search_config.clone(),
            dense: cfg.dense,
            lexical: cfg.lexical,
            min_score: cfg.min_score,
            collapse_tree: cfg.collapse_tree,
        }
    }
}

/// Hybrid retrieval over the chunks table.
pub struct PgRetriever {
    pool: PgPool,
    embedder: Arc<dyn Embedder>,
    tuning: RetrievalTuning,
}

impl PgRetriever {
    /// Builds a retriever.
    pub fn new(pool: PgPool, embedder: Arc<dyn Embedder>, tuning: RetrievalTuning) -> Self {
        Self {
            pool,
            embedder,
            tuning,
        }
    }

    /// The categories the scraper has published, sorted. Empty when nothing is indexed.
    ///
    /// # Errors
    /// Returns [RetrievalError::Store] when the query fails.
    pub async fn categories(&self) -> Result<Vec<String>, RetrievalError> {
        sqlx::query_scalar("select distinct category from sources order by category")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| RetrievalError::Store(e.to_string()))
    }
}

#[derive(Clone)]
struct Candidate {
    chunk_id: Uuid,
    /// The summary this row was folded into, when it has one.
    parent_id: Option<Uuid>,
    source_id: Uuid,
    title: String,
    url: Option<String>,
    content: String,
    fetched_at: DateTime<Utc>,
}

fn row_to_candidate(row: &sqlx::postgres::PgRow) -> Result<Candidate, sqlx::Error> {
    Ok(Candidate {
        chunk_id: row.try_get("chunk_id")?,
        parent_id: row.try_get("parent_id")?,
        source_id: row.try_get("source_id")?,
        title: row.try_get("title")?,
        url: row.try_get("url")?,
        content: row.try_get("content")?,
        fetched_at: row.try_get("fetched_at")?,
    })
}

/// Public ASU content is written under tenant public and visible to every guild.
const SELECT: &str =
    "select c.id as chunk_id, c.source_id, s.key as title, s.url, c.content, c.fetched_at,
            c.parent_id
    from chunks c join sources s on s.id = c.source_id
    where (c.tenant_id = $1 or c.tenant_id = 'public')
      and (cardinality($2::text[]) = 0 or c.category = any($2))";

/// Drops a row whose summary is already in the result, keeping the higher ranked of the two.
///
/// Fused order is best first, so the first mention of a pair is the one kept.
pub(crate) fn collapse(rows: &[(Uuid, Option<Uuid>)]) -> Vec<bool> {
    let mut seen: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    rows.iter()
        .map(|(id, parent)| {
            if parent.is_some_and(|p| seen.contains(&p)) {
                return false;
            }
            seen.insert(*id);
            true
        })
        .collect()
}

/// Reciprocal rank fusion. Each ranked list contributes 1 / (k + rank).
pub(crate) fn rrf(lists: &[Vec<Uuid>], k: f32) -> Vec<(Uuid, f32)> {
    let mut scores: HashMap<Uuid, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            // Ranks are small; the cast cannot lose precision.
            #[allow(clippy::cast_precision_loss)]
            let contribution = 1.0 / (k + rank as f32 + 1.0);
            *scores.entry(*id).or_insert(0.0) += contribution;
        }
    }
    let mut fused: Vec<(Uuid, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused
}

#[async_trait]
impl Retriever for PgRetriever {
    async fn retrieve(
        &self,
        ctx: &RequestContext,
        query: &RetrievalQuery,
    ) -> Result<Vec<Evidence>, RetrievalError> {
        let store = |e: sqlx::Error| RetrievalError::Store(e.to_string());
        let mut by_id: HashMap<Uuid, Candidate> = HashMap::new();
        let mut ranked: Vec<Vec<Uuid>> = Vec::with_capacity(2);

        if self.tuning.dense {
            let vectors = self
                .embedder
                .embed(std::slice::from_ref(&query.text))
                .await?;
            let Some(vector) = vectors.into_iter().next() else {
                return Err(RetrievalError::Embedding("no vector returned".into()));
            };
            if vector.len() != self.embedder.dim() {
                return Err(RetrievalError::Embedding(format!(
                    "embedding has {} dimensions; the index holds {}",
                    vector.len(),
                    self.embedder.dim()
                )));
            }
            let sql = format!("{SELECT} order by c.embedding <=> $3::vector limit $4");
            let rows = sqlx::query(&sql)
                .bind(&ctx.tenant_id)
                .bind(&query.categories)
                .bind(vector_literal(&vector))
                .bind(self.tuning.candidates)
                .fetch_all(&self.pool)
                .await
                .map_err(store)?;
            let mut ids = Vec::with_capacity(rows.len());
            for row in &rows {
                let c = row_to_candidate(row).map_err(store)?;
                ids.push(c.chunk_id);
                by_id.insert(c.chunk_id, c);
            }
            ranked.push(ids);
        }

        if self.tuning.lexical {
            // The text search configuration names a Postgres object and cannot be bound as a
            // parameter. It is quoted as a literal and validated at load.
            let cfg = quote_literal(&self.tuning.text_search_config);
            let sql = format!(
                "{SELECT} and c.tsv @@ websearch_to_tsquery({cfg}, $3)
                 order by ts_rank_cd(c.tsv, websearch_to_tsquery({cfg}, $3)) desc limit $4"
            );
            let rows = sqlx::query(&sql)
                .bind(&ctx.tenant_id)
                .bind(&query.categories)
                .bind(&query.text)
                .bind(self.tuning.candidates)
                .fetch_all(&self.pool)
                .await
                .map_err(store)?;
            let mut ids = Vec::with_capacity(rows.len());
            for row in &rows {
                let c = row_to_candidate(row).map_err(store)?;
                ids.push(c.chunk_id);
                by_id.entry(c.chunk_id).or_insert(c);
            }
            ranked.push(ids);
        }

        if by_id.is_empty() {
            return Ok(Vec::new());
        }

        let fused = rrf(&ranked, self.tuning.rrf_k);
        let ordered: Vec<(Candidate, f32)> = fused
            .into_iter()
            .filter(|(_, score)| *score >= self.tuning.min_score)
            .filter_map(|(id, score)| by_id.remove(&id).map(|c| (c, score)))
            .collect();
        let ordered = if self.tuning.collapse_tree {
            let rows: Vec<(Uuid, Option<Uuid>)> = ordered
                .iter()
                .map(|(c, _)| (c.chunk_id, c.parent_id))
                .collect();
            let keep = collapse(&rows);
            ordered
                .into_iter()
                .zip(keep)
                .filter_map(|(row, keep)| keep.then_some(row))
                .collect()
        } else {
            ordered
        };

        Ok(ordered
            .into_iter()
            .take(query.top_k)
            .map(|(c, score)| Evidence {
                source_id: c.source_id,
                chunk_id: c.chunk_id,
                title: c.title,
                content: c.content,
                url: c.url,
                fetched_at: c.fetched_at,
                score,
            })
            .collect())
    }
}
