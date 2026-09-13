//! SourceQueries trait: the registry of live query sources and the queue that runs them.

use async_trait::async_trait;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::query::{
    QueryError, QueryOutcome, QueryRequest, QuerySourceInfo,
};

/// Runs a parameterized source query, reports what the scraper published; engine never fetches.
#[async_trait]
pub trait SourceQueries: Send + Sync {
    /// Sources currently offered. Read once at boot to build the tool.
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError>;

    /// Queues request and waits for the scraper answer, or fails inside the request budget.
    async fn run(
        &self,
        ctx: &RequestContext,
        request: &QueryRequest,
    ) -> Result<QueryOutcome, QueryError>;
}
