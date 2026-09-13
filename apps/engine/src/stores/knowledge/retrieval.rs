//! Hybrid retrieval over the chunks table: pgvector dense and full-text lexical, fused with RRF.

use std::collections::{HashMap, HashSet};
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

/// How the two retrieval legs run and fuse. Built from the retrieval section; there is no Default.
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
    /// Drop dense matches farther than this cosine distance.
    pub max_distance: f32,
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
            max_distance: cfg.max_distance,
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

/// The candidate columns of a chunk.
const COLUMNS: &str = "c.id as chunk_id, c.source_id, s.key as title, s.url, c.content, \
                       c.fetched_at, c.parent_id";

/// Chunks of the caller tenant and of tenant public, which every guild reads.
const FROM: &str = "from chunks c join sources s on s.id = c.source_id
    where (c.tenant_id = $1 or c.tenant_id = 'public')";

/// Drops a row whose summary is already in the result, keeping the higher ranked of the two.
pub(crate) fn collapse(rows: &[(Uuid, Option<Uuid>)]) -> Vec<bool> {
    let mut seen: HashSet<Uuid> = HashSet::new();
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
            // Ranks are small enough for an exact f32.
            #[allow(clippy::cast_precision_loss)]
            let contribution = 1.0 / (k + rank as f32 + 1.0);
            *scores.entry(*id).or_insert(0.0) += contribution;
        }
    }
    let mut fused: Vec<(Uuid, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused
}

/// Maps a sqlx error to a RetrievalError.
#[allow(clippy::needless_pass_by_value)]
fn store(e: sqlx::Error) -> RetrievalError {
    RetrievalError::Store(e.to_string())
}

/// Adds the candidates of one leg to by_id, first seen wins, and returns their ids in rank order.
fn rank(
    rows: &[sqlx::postgres::PgRow],
    by_id: &mut HashMap<Uuid, Candidate>,
) -> Result<Vec<Uuid>, RetrievalError> {
    let mut ids = Vec::with_capacity(rows.len());
    for row in rows {
        let c = row_to_candidate(row).map_err(store)?;
        ids.push(c.chunk_id);
        by_id.entry(c.chunk_id).or_insert(c);
    }
    Ok(ids)
}

impl PgRetriever {
    /// The pgvector leg: the nearest chunks to the embedded query.
    async fn dense_rows(
        &self,
        ctx: &RequestContext,
        query: &RetrievalQuery,
    ) -> Result<Vec<sqlx::postgres::PgRow>, RetrievalError> {
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
        let sql = format!(
            "select * from (
                select {COLUMNS}, c.embedding <=> $2::vector as distance {FROM}
                order by distance limit $3
            ) nearest where distance <= $4"
        );
        sqlx::query(&sql)
            .bind(&ctx.tenant_id)
            .bind(vector_literal(&vector))
            .bind(self.tuning.candidates)
            .bind(f64::from(self.tuning.max_distance))
            .fetch_all(&self.pool)
            .await
            .map_err(store)
    }

    /// The full-text leg: chunks matching the query, best ts_rank_cd first.
    async fn lexical_rows(
        &self,
        ctx: &RequestContext,
        query: &RetrievalQuery,
    ) -> Result<Vec<sqlx::postgres::PgRow>, RetrievalError> {
        // The text search configuration is validated at load and quoted as a literal.
        let cfg = quote_literal(&self.tuning.text_search_config);
        let sql = format!(
            "select {COLUMNS} {FROM} and c.tsv @@ websearch_to_tsquery({cfg}, $2)
             order by ts_rank_cd(c.tsv, websearch_to_tsquery({cfg}, $2)) desc limit $3"
        );
        sqlx::query(&sql)
            .bind(&ctx.tenant_id)
            .bind(&query.text)
            .bind(self.tuning.candidates)
            .fetch_all(&self.pool)
            .await
            .map_err(store)
    }
}

#[async_trait]
impl Retriever for PgRetriever {
    async fn retrieve(
        &self,
        ctx: &RequestContext,
        query: &RetrievalQuery,
    ) -> Result<Vec<Evidence>, RetrievalError> {
        let mut by_id: HashMap<Uuid, Candidate> = HashMap::new();
        let mut ranked: Vec<Vec<Uuid>> = Vec::with_capacity(2);
        if self.tuning.dense {
            let rows = self.dense_rows(ctx, query).await?;
            ranked.push(rank(&rows, &mut by_id)?);
        }
        if self.tuning.lexical {
            let rows = self.lexical_rows(ctx, query).await?;
            ranked.push(rank(&rows, &mut by_id)?);
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
