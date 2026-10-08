//! Live queries on the platform ASU module: its registry, and a fetch it runs while the call waits.

use async_trait::async_trait;
use reqwest::Method;
use serde_json::json;

use super::client::{Call, PlatformClient, PlatformError, field};
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::query::{
    QueryError, QueryOutcome, QueryRequest, QuerySourceInfo,
};

/// Live queries run by the platform.
pub struct PlatformQueries {
    client: PlatformClient,
}

impl PlatformQueries {
    /// Wraps a client.
    pub fn new(client: PlatformClient) -> Self {
        Self { client }
    }
}

/// Maps a platform error: a refusal of the query goes back to the model, the rest are failures.
fn query_error(e: PlatformError) -> QueryError {
    match e {
        PlatformError::Timeout(budget) => QueryError::Timeout(budget),
        PlatformError::Status {
            status: 400 | 404 | 422 | 502 | 503,
            message,
        } => QueryError::Rejected(message),
        other => QueryError::Store(other.to_string()),
    }
}

#[async_trait]
impl SourceQueries for PlatformQueries {
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError> {
        let value = self
            .client
            .call(&Call::new(Method::GET, "/api/asu/queries"))
            .await
            .map_err(query_error)?;
        field(&value, "queries").map_err(|e| QueryError::Store(e.to_string()))
    }

    async fn run(
        &self,
        ctx: &RequestContext,
        request: &QueryRequest,
    ) -> Result<QueryOutcome, QueryError> {
        let remaining = ctx.remaining();
        if remaining.is_zero() {
            return Err(QueryError::Timeout(remaining));
        }
        let body = json!({ "source": request.source, "params": request.params });
        let call = Call::new(Method::POST, "/api/asu/query")
            .body(&body)
            .timeout(remaining);
        let value = tokio::select! {
            () = ctx.cancel.cancelled() => return Err(QueryError::Cancelled),
            answer = self.client.call(&call) => answer.map_err(query_error)?,
        };
        serde_json::from_value(value).map_err(|e| QueryError::Store(format!("query result: {e}")))
    }
}
