//! SourceQueries trait: the registry of live query sources and the queue that runs them.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::query::{QueryError, QueryOutcome, QueryRequest, QuerySourceInfo};

/// Runs a parameterized source query and reports what the scraper published.
///
/// The engine never fetches the page itself: the scraper owns fetching, and the two meet only
/// in the database.
#[async_trait]
pub trait SourceQueries: Send + Sync {
    /// Sources currently offered. Read once at boot to build the tool.
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError>;

    /// Queues request and waits for the worker answer, or fails inside the request budget.
    async fn run(
        &self,
        ctx: &RequestContext,
        request: &QueryRequest,
    ) -> Result<QueryOutcome, QueryError>;
}
