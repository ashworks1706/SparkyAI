//! Source query registry and the jobs queue the scraper serves.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use sqlx::Row;
use sqlx::postgres::PgPool;
use sqlx::postgres::types::PgInterval;
use uuid::Uuid;

use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::query::{
    QueryError, QueryOutcome, QueryParam, QueryRequest, QuerySourceInfo,
};

/// The registry and job queue the scraper serves.
///
/// The engine writes a jobs row and waits for the answer to appear on it.
pub struct PgSourceQueries {
    pool: PgPool,
    poll: Duration,
    claim: Duration,
}

/// The jobs.kind the scraper claims.
const QUERY_JOB_KIND: &str = "source_query";

/// The channel the scraper listens on for queued jobs.
const QUERY_CHANNEL: &str = "source_query";

impl PgSourceQueries {
    /// Polls a queued job every poll interval until it resolves or the request runs out of time.
    /// A job the scraper does not claim within claim reports that it is not running.
    pub fn new(pool: PgPool, interval: Duration, claim: Duration) -> Self {
        Self {
            pool,
            poll: interval,
            claim,
        }
    }

    /// Marks a job cancelled so the scraper drops it if it has not started it.
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

/// A query deadline as a Postgres interval, kept to the microseconds an interval holds.
///
/// # Errors
/// Returns [QueryError::Store] when the deadline does not fit an interval.
pub(crate) fn deadline_interval(remaining: Duration) -> Result<PgInterval, QueryError> {
    let micros = u64::try_from(remaining.as_micros()).unwrap_or(u64::MAX);
    PgInterval::try_from(Duration::from_micros(micros)).map_err(|e| {
        QueryError::Store(format!(
            "deadline {remaining:?} is not a valid interval: {e}"
        ))
    })
}

#[async_trait]
impl SourceQueries for PgSourceQueries {
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError> {
        let store = |e: sqlx::Error| QueryError::Store(e.to_string());
        let rows = sqlx::query("select key, params from query_sources where enabled order by key")
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
        .bind(deadline_interval(remaining)?)
        .fetch_one(&self.pool)
        .await
        .map_err(store)?;
        // The scraper wakes on the notification; without one it finds the job on its next poll.
        sqlx::query("select pg_notify($1, $2)")
            .bind(QUERY_CHANNEL)
            .bind(job_id.to_string())
            .execute(&self.pool)
            .await
            .map_err(store)?;

        let queued = std::time::Instant::now();
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
                        error.unwrap_or_else(|| "the scraper gave no reason".into()),
                    ));
                }
                "cancelled" => return Err(QueryError::Cancelled),
                "queued" if queued.elapsed() >= self.claim => {
                    self.cancel(job_id).await;
                    return Err(QueryError::NoWorker(request.source.clone()));
                }
                "queued" | "running" => {}
                other => {
                    self.cancel(job_id).await;
                    return Err(QueryError::Store(format!(
                        "job {job_id} has unknown status {other}"
                    )));
                }
            }
            // The sleep never runs past the deadline.
            tokio::time::sleep(self.poll.min(ctx.remaining())).await;
        }
    }
}
