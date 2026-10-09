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
use crate::stores::standalone::knowledge::window::{Span, Take, collapse, rrf, spans, takes};
use crate::stores::standalone::postgres::{quote_literal, vector_literal};

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
    /// Rows either side of a hit read back with it. 0 hands back the hit alone.
    pub window: i32,
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
            window: cfg.window,
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
    /// Registry key of the source.
    key: String,
    /// The summary this row was folded into, when it has one.
    parent_id: Option<Uuid>,
    source_id: Uuid,
    title: String,
    url: Option<String>,
    content: String,
    fetched_at: DateTime<Utc>,
    /// The source version the row belongs to. Rows of one version are contiguous by ordinal.
    version_id: Uuid,
    /// Position within the version.
    ordinal: i32,
    /// 0 for a chunk of the page, higher for a summary over chunks.
    level: i32,
}

fn row_to_candidate(row: &sqlx::postgres::PgRow) -> Result<Candidate, sqlx::Error> {
    Ok(Candidate {
        chunk_id: row.try_get("chunk_id")?,
        key: row.try_get("key")?,
        parent_id: row.try_get("parent_id")?,
        source_id: row.try_get("source_id")?,
        title: row.try_get("title")?,
        url: row.try_get("url")?,
        content: row.try_get("content")?,
        fetched_at: row.try_get("fetched_at")?,
        version_id: row.try_get("version_id")?,
        ordinal: row.try_get("ordinal")?,
        level: row.try_get("level")?,
    })
}

/// The candidate columns of a chunk. A source without a stored page title falls back to its key.
const COLUMNS: &str = "c.id as chunk_id, c.source_id, s.key, \
                       coalesce(nullif(s.title, ''), s.key) as title, s.url, c.content, \
                       c.fetched_at, c.parent_id, c.version_id, c.ordinal, c.level";

/// Chunks of the caller tenant and of tenant public, which every guild reads.
fn from(at: usize) -> String {
    format!(
        "from chunks c join sources s on s.id = c.source_id
    where (c.tenant_id = $1 or c.tenant_id = 'public')
      and (${at} = '' or c.category = ${at})"
    )
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
        let scoped = from(5);
        let sql = format!(
            "select * from (
                select {COLUMNS}, c.embedding <=> $2::vector as distance {scoped}
                order by distance limit $3
            ) nearest where distance <= $4"
        );
        // Only constants, placeholder numbers and the quoted, validated config are interpolated.
        sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
            .bind(&ctx.tenant_id)
            .bind(vector_literal(&vector))
            .bind(self.tuning.candidates)
            .bind(f64::from(self.tuning.max_distance))
            .bind(query.category.clone().unwrap_or_default())
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
        let scoped = from(4);
        let sql = format!(
            "select {COLUMNS} {scoped} and c.tsv @@ websearch_to_tsquery({cfg}, $2)
             order by ts_rank_cd(c.tsv, websearch_to_tsquery({cfg}, $2)) desc limit $3"
        );
        // Only constants, placeholder numbers and the quoted, validated config are interpolated.
        sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
            .bind(&ctx.tenant_id)
            .bind(&query.text)
            .bind(self.tuning.candidates)
            .bind(query.category.clone().unwrap_or_default())
            .fetch_all(&self.pool)
            .await
            .map_err(store)
    }
}

impl PgRetriever {
    /// The text of each span, its rows joined in order, aligned with the spans given.
    async fn windows(
        &self,
        ctx: &RequestContext,
        spans: &[Span],
    ) -> Result<Vec<String>, RetrievalError> {
        if spans.is_empty() {
            return Ok(Vec::new());
        }
        let versions: Vec<Uuid> = spans.iter().map(|s| s.version_id).collect();
        let los: Vec<i32> = spans.iter().map(|s| s.lo).collect();
        let his: Vec<i32> = spans.iter().map(|s| s.hi).collect();
        let sql = "select c.version_id, c.ordinal, c.content
             from unnest($2::uuid[], $3::int4[], $4::int4[]) as w(version_id, lo, hi)
             join chunks c
               on c.version_id = w.version_id
              and c.ordinal between w.lo and w.hi
              and c.level = 0
             where (c.tenant_id = $1 or c.tenant_id = 'public')";
        let rows = sqlx::query(sql)
            .bind(&ctx.tenant_id)
            .bind(&versions)
            .bind(&los)
            .bind(&his)
            .fetch_all(&self.pool)
            .await
            .map_err(store)?;

        let mut by_row: HashMap<(Uuid, i32), String> = HashMap::new();
        for row in &rows {
            let version_id: Uuid = row.try_get("version_id").map_err(store)?;
            let ordinal: i32 = row.try_get("ordinal").map_err(store)?;
            let content: String = row.try_get("content").map_err(store)?;
            by_row.insert((version_id, ordinal), content);
        }
        Ok(spans.iter().map(|span| span.join(&by_row)).collect())
    }

    /// Each hit with the passage around it, or its own text when it is a summary or stands alone.
    async fn widened(
        &self,
        ctx: &RequestContext,
        ordered: Vec<(Candidate, f32)>,
    ) -> Result<Vec<Evidence>, RetrievalError> {
        let leaves: Vec<(Uuid, i32)> = ordered
            .iter()
            .filter(|(c, _)| c.level == 0)
            .map(|(c, _)| (c.version_id, c.ordinal))
            .collect();
        let spans = spans(&leaves, self.tuning.window);
        let widened = self.windows(ctx, &spans).await?;

        let rows: Vec<(i32, Uuid, i32)> = ordered
            .iter()
            .map(|(c, _)| (c.level, c.version_id, c.ordinal))
            .collect();
        let mut evidence = Vec::with_capacity(ordered.len());
        for ((c, score), take) in ordered.into_iter().zip(takes(&rows, &spans)) {
            let content = match take {
                Take::Skip => continue,
                Take::Own => c.content,
                // A span with no rows behind it falls back to the text of the hit.
                Take::Span(at) => match widened.get(at) {
                    Some(text) if !text.is_empty() => text.clone(),
                    _ => c.content,
                },
            };
            evidence.push(Evidence {
                source_id: c.source_id,
                chunk_id: c.chunk_id,
                key: c.key,
                title: c.title,
                content,
                url: c.url,
                fetched_at: c.fetched_at,
                score,
            });
        }
        Ok(evidence)
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

        let top: Vec<(Candidate, f32)> = ordered.into_iter().take(query.top_k).collect();
        if self.tuning.window <= 0 {
            return Ok(top
                .into_iter()
                .map(|(c, score)| Evidence {
                    source_id: c.source_id,
                    chunk_id: c.chunk_id,
                    key: c.key,
                    title: c.title,
                    content: c.content,
                    url: c.url,
                    fetched_at: c.fetched_at,
                    score,
                })
                .collect());
        }
        self.widened(ctx, top).await
    }
}
