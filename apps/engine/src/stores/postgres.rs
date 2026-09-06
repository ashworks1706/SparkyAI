//! `PostgreSQL` adapter: `Retriever` (pgvector dense + FTS lexical, fused with RRF),
//! `ConversationStore`, and `MemoryStore` over one connection pool.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;
use sqlx::Row;
use sqlx::postgres::{PgPool, PgPoolOptions};
use uuid::Uuid;

use crate::core::traits::confirmation::ConfirmationStore;
use crate::core::traits::conversation::ConversationStore;
use crate::core::traits::memory::MemoryStore;
use crate::core::traits::query::SourceQueries;
use crate::core::traits::retrieval::{Embedder, Retriever};
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::memory::{Memory, MemoryKind, MemoryQuery};
use crate::core::types::message::Message;
use crate::core::types::policy::PendingAction;
use crate::core::types::query::{
    QueryError, QueryOutcome, QueryParam, QueryRequest, QuerySourceInfo,
};
use crate::core::types::retrieval::{RetrievalError, RetrievalQuery};
use crate::core::types::store::StoreError;

/// Opens the pool. Fails fast if the database is unreachable.
pub async fn connect(
    url: &SecretString,
    max_connections: u32,
    acquire_timeout: Duration,
) -> Result<PgPool, StoreError> {
    PgPoolOptions::new()
        .max_connections(max_connections)
        .acquire_timeout(acquire_timeout)
        .connect(url.expose_secret())
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
}

/// Used as a `map_err` function pointer, so it takes the error by value.
#[allow(clippy::needless_pass_by_value)]
fn db(e: sqlx::Error) -> StoreError {
    StoreError::Database(e.to_string())
}

/// How the two retrieval legs are run and fused.
#[derive(Debug, Clone)]
pub struct RetrievalTuning {
    /// Candidates pulled from each leg before fusion.
    pub candidates: i64,
    /// Reciprocal rank fusion constant.
    pub rrf_k: f32,
    /// `PostgreSQL` text search configuration for the lexical leg.
    pub text_search_config: String,
    /// Run the pgvector leg.
    pub dense: bool,
    /// Run the full-text leg.
    pub lexical: bool,
    /// Drop fused results below this score.
    pub min_score: f32,
}

impl Default for RetrievalTuning {
    fn default() -> Self {
        Self {
            candidates: 20,
            rrf_k: 60.0,
            text_search_config: "english".into(),
            dense: true,
            lexical: true,
            min_score: 0.0,
        }
    }
}

/// Hybrid retrieval over the `chunks` table.
pub struct PgRetriever {
    pool: PgPool,
    embedder: Arc<dyn Embedder>,
    tuning: RetrievalTuning,
}

impl PgRetriever {
    /// Builds a retriever. Fused order is final.
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
    source_id: Uuid,
    title: String,
    url: Option<String>,
    content: String,
    fetched_at: DateTime<Utc>,
}

fn row_to_candidate(row: &sqlx::postgres::PgRow) -> Result<Candidate, sqlx::Error> {
    Ok(Candidate {
        chunk_id: row.try_get("chunk_id")?,
        source_id: row.try_get("source_id")?,
        title: row.try_get("title")?,
        url: row.try_get("url")?,
        content: row.try_get("content")?,
        fetched_at: row.try_get("fetched_at")?,
    })
}

/// Public ASU content is written under tenant `public` and visible to every guild.
const SELECT: &str =
    "select c.id as chunk_id, c.source_id, s.key as title, s.url, c.content, c.fetched_at
    from chunks c join sources s on s.id = c.source_id
    where (c.tenant_id = $1 or c.tenant_id = 'public')
      and (cardinality($2::text[]) = 0 or c.category = any($2))";

/// A `PostgreSQL` string literal. Used for the text search configuration, which names an
/// object and so cannot be a bind parameter.
pub(crate) fn quote_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// pgvector's text input form: `[0.1,0.2,...]`.
pub(crate) fn vector_literal(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 10 + 2);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&x.to_string());
    }
    s.push(']');
    s
}

/// Reciprocal rank fusion: each ranked list contributes 1 / (k + rank).
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
            // The text search configuration names a Postgres object, so it cannot be bound as
            // a parameter. It is quoted as a literal, which is why it is validated at load.
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

/// Conversations and messages tables.
pub struct PgConversations {
    pool: PgPool,
}

impl PgConversations {
    /// Wraps a pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    async fn user_row_id(&self, ctx: &RequestContext) -> Result<Uuid, StoreError> {
        let row = sqlx::query(
            "insert into users (tenant_id, discord_id, roles) values ($1, $2, $3)
             on conflict (tenant_id, discord_id) do update set roles = excluded.roles
             returning id",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(&ctx.roles)
        .fetch_one(&self.pool)
        .await
        .map_err(db)?;
        row.try_get("id").map_err(db)
    }
}

#[async_trait]
impl ConversationStore for PgConversations {
    async fn ensure(&self, ctx: &RequestContext, channel_id: &str) -> Result<(), StoreError> {
        let user_id = self.user_row_id(ctx).await?;
        sqlx::query(
            "insert into conversations (id, tenant_id, user_id, channel_id) values ($1, $2, $3, $4)
             on conflict (id) do nothing",
        )
        .bind(ctx.conversation_id)
        .bind(&ctx.tenant_id)
        .bind(user_id)
        .bind(channel_id)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn load(&self, ctx: &RequestContext, limit: usize) -> Result<Vec<Message>, StoreError> {
        let rows = sqlx::query(
            "select m.content from messages m join conversations c on c.id = m.conversation_id
             where m.conversation_id = $1 and c.tenant_id = $2
             order by m.created_at desc limit $3",
        )
        .bind(ctx.conversation_id)
        .bind(&ctx.tenant_id)
        .bind(i64::try_from(limit).unwrap_or(i64::MAX))
        .fetch_all(&self.pool)
        .await
        .map_err(db)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows.iter().rev() {
            let value: serde_json::Value = row.try_get("content").map_err(db)?;
            let message = serde_json::from_value::<Message>(value)
                .map_err(|e| StoreError::Database(format!("stored message unreadable: {e}")))?;
            out.push(message);
        }
        Ok(out)
    }

    async fn append(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await.map_err(db)?;
        for m in turns {
            let content =
                serde_json::to_value(m).map_err(|e| StoreError::Database(e.to_string()))?;
            sqlx::query(
                "insert into messages (conversation_id, role, content) values ($1, $2, $3)",
            )
            .bind(ctx.conversation_id)
            .bind(role_str(m))
            .bind(content)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        }
        sqlx::query("update conversations set updated_at = now() where id = $1")
            .bind(ctx.conversation_id)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        tx.commit().await.map_err(db)
    }
}

fn role_str(m: &Message) -> &'static str {
    match m.role {
        crate::core::types::message::Role::System => "system",
        crate::core::types::message::Role::User => "user",
        crate::core::types::message::Role::Assistant => "assistant",
        crate::core::types::message::Role::Tool => "tool",
    }
}

/// Memories table. Every query is scoped by tenant and user; the interface cannot cross users.
pub struct PgMemory {
    pool: PgPool,
}

impl PgMemory {
    /// Wraps a pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl MemoryStore for PgMemory {
    async fn recall(
        &self,
        ctx: &RequestContext,
        q: &MemoryQuery,
    ) -> Result<Vec<Memory>, StoreError> {
        let kinds: Vec<&str> = q.kinds.iter().map(|k| k.as_str()).collect();
        let rows = sqlx::query(
            "select m.id, m.kind, m.content, m.confidence, m.created_at, m.expires_at
             from memories m join users u on u.id = m.user_id
             where m.tenant_id = $1 and u.discord_id = $2
               and (cardinality($3::text[]) = 0 or m.kind = any($3))
               and (m.expires_at is null or m.expires_at > now())
             order by m.confidence desc, m.created_at desc limit $4",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(&kinds)
        .bind(i64::try_from(q.limit).unwrap_or(i64::MAX))
        .fetch_all(&self.pool)
        .await
        .map_err(db)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let kind: String = row.try_get("kind").map_err(db)?;
            let Some(kind) = MemoryKind::parse(&kind) else {
                return Err(StoreError::Database(format!(
                    "memories.kind {kind:?} is not a kind this build knows; schema and code \
                     disagree"
                )));
            };
            out.push(Memory {
                id: row.try_get("id").map_err(db)?,
                kind,
                content: row.try_get("content").map_err(db)?,
                confidence: row.try_get("confidence").map_err(db)?,
                created_at: row.try_get("created_at").map_err(db)?,
                expires_at: row.try_get("expires_at").map_err(db)?,
            });
        }
        Ok(out)
    }
}

/// Actions waiting on their caller's approval, in `confirmations`.
pub struct PgConfirmations {
    pool: PgPool,
}

impl PgConfirmations {
    /// Wraps a pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl ConfirmationStore for PgConfirmations {
    async fn hold(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        pending: &PendingAction,
        payload_hash: &str,
        ttl: Duration,
    ) -> Result<(), StoreError> {
        let action =
            serde_json::to_value(pending).map_err(|e| StoreError::Database(e.to_string()))?;
        let expires_at = Utc::now()
            + chrono::Duration::from_std(ttl)
                .map_err(|e| StoreError::Database(format!("confirmation ttl: {e}")))?;
        sqlx::query(
            "insert into confirmations
               (id, request_id, user_id, action, payload_hash, status, expires_at)
             select $1, $2, u.id, $3, $4, 'pending', $5
             from users u where u.tenant_id = $6 and u.discord_id = $7",
        )
        .bind(token)
        .bind(ctx.request_id)
        .bind(&action)
        .bind(payload_hash)
        .bind(expires_at)
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn claim(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        approved: bool,
    ) -> Result<Option<PendingAction>, StoreError> {
        // One statement: the row moves out of `pending` as it is read, so a second click and a
        // second caller both find nothing. The user join is what limits it to whoever asked.
        let row = sqlx::query(
            "update confirmations c
                set status = case when $1 then 'confirmed' else 'denied' end,
                    resolved_at = now()
              from users u
             where c.id = $2
               and c.user_id = u.id
               and u.tenant_id = $3
               and u.discord_id = $4
               and c.status = 'pending'
               and c.expires_at > now()
            returning c.action",
        )
        .bind(approved)
        .bind(token)
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(db)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let action: serde_json::Value = row.try_get("action").map_err(db)?;
        serde_json::from_value(action)
            .map(Some)
            .map_err(|e| StoreError::Database(format!("held action unreadable: {e}")))
    }
}

/// The registry and job queue the scraper's worker serves.
///
/// The engine never fetches the page: it writes a `jobs` row and waits for the answer to appear
/// on it. That is the whole channel between the two processes.
pub struct PgSourceQueries {
    pool: PgPool,
    poll: Duration,
}

/// `jobs.kind` the scraper's worker claims.
const QUERY_JOB_KIND: &str = "source_query";

impl PgSourceQueries {
    /// Polls a queued job every `poll` until it resolves or the request runs out of time.
    pub fn new(pool: PgPool, interval: Duration) -> Self {
        Self {
            pool,
            poll: interval,
        }
    }

    /// Marks a job cancelled so a worker that has not started it drops it.
    async fn cancel(&self, job_id: Uuid) {
        let result = sqlx::query(
            "update jobs set status = 'cancelled', updated_at = now()
             where id = $1 and status in ('queued', 'running')",
        )
        .bind(job_id)
        .execute(&self.pool)
        .await;
        if let Err(e) = result {
            tracing::warn!(error = %e, %job_id, "could not cancel query job");
        }
    }
}

#[async_trait]
impl SourceQueries for PgSourceQueries {
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError> {
        let store = |e: sqlx::Error| QueryError::Store(e.to_string());
        let rows = sqlx::query(
            "select key, description, params from query_sources where enabled order by key",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(store)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let params: Value = row.try_get("params").map_err(store)?;
            let params: Vec<QueryParam> = serde_json::from_value(params)
                .map_err(|e| QueryError::Store(format!("query_sources.params: {e}")))?;
            out.push(QuerySourceInfo {
                key: row.try_get("key").map_err(store)?,
                description: row.try_get("description").map_err(store)?,
                params,
            });
        }
        Ok(out)
    }

    async fn run(
        &self,
        ctx: &RequestContext,
        request: &QueryRequest,
    ) -> Result<QueryOutcome, QueryError> {
        let store = |e: sqlx::Error| QueryError::Store(e.to_string());
        let remaining = ctx.remaining();
        let input = serde_json::json!({ "source": request.source, "params": request.params });
        let job_id: Uuid = sqlx::query_scalar(
            "insert into jobs (kind, status, owner, input, deadline)
             values ($1, 'queued', $2, $3, now() + $4::interval)
             returning id",
        )
        .bind(QUERY_JOB_KIND)
        .bind(&ctx.user_id)
        .bind(&input)
        .bind(
            sqlx::postgres::types::PgInterval::try_from(remaining).map_err(|e| {
                QueryError::Store(format!(
                    "deadline {remaining:?} is not a valid interval: {e}"
                ))
            })?,
        )
        .fetch_one(&self.pool)
        .await
        .map_err(store)?;

        loop {
            if ctx.cancel.is_cancelled() {
                self.cancel(job_id).await;
                return Err(QueryError::Cancelled);
            }
            if ctx.remaining().is_zero() {
                self.cancel(job_id).await;
                return Err(QueryError::Timeout(remaining));
            }
            let row = sqlx::query("select status, result, error from jobs where id = $1")
                .bind(job_id)
                .fetch_one(&self.pool)
                .await
                .map_err(store)?;
            let status: String = row.try_get("status").map_err(store)?;
            match status.as_str() {
                "done" => {
                    let result: Option<Value> = row.try_get("result").map_err(store)?;
                    let Some(result) = result else {
                        return Err(QueryError::Store("job finished with no result".into()));
                    };
                    return serde_json::from_value(result)
                        .map_err(|e| QueryError::Store(format!("job result: {e}")));
                }
                "failed" => {
                    let error: Option<String> = row.try_get("error").map_err(store)?;
                    return Err(QueryError::Rejected(
                        error.unwrap_or_else(|| "the source worker gave no reason".into()),
                    ));
                }
                "cancelled" => return Err(QueryError::Cancelled),
                _ => {}
            }
            // Sleeping past the deadline would report a timeout later than it happened.
            tokio::time::sleep(self.poll.min(ctx.remaining())).await;
        }
    }
}
