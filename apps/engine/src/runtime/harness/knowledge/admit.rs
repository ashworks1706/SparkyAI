//! The cap on live queries in flight, so a spike sheds load instead of saturating the pool.

use std::sync::Arc;

use async_trait::async_trait;

use crate::core::traits::knowledge::admission::Admission;
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::query::{
    QueryError, QueryOutcome, QueryRequest, QuerySourceInfo,
};
use crate::core::types::trace::TraceEvent;

/// Live queries under a cap on how many run at once. A query over the cap is refused, not queued.
pub struct AdmittedQueries {
    inner: Arc<dyn SourceQueries>,
    admission: Arc<dyn Admission>,
    trace: Arc<dyn TraceSink>,
}

impl AdmittedQueries {
    /// Wraps queries in the cap.
    pub fn new(
        inner: Arc<dyn SourceQueries>,
        admission: Arc<dyn Admission>,
        trace: Arc<dyn TraceSink>,
    ) -> Self {
        Self {
            inner,
            admission,
            trace,
        }
    }
}

#[async_trait]
impl SourceQueries for AdmittedQueries {
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError> {
        self.inner.sources().await
    }

    async fn run(
        &self,
        ctx: &RequestContext,
        request: &QueryRequest,
    ) -> Result<QueryOutcome, QueryError> {
        let holder = ctx.request_id.to_string();
        match self.admission.enter(&holder).await {
            Ok(true) => {}
            Ok(false) => {
                self.trace.emit(
                    ctx,
                    TraceEvent::QueryRefused {
                        source: request.source.clone(),
                    },
                );
                return Err(QueryError::Busy(request.source.clone()));
            }
            // A cap that cannot be read caps nothing; the query still runs.
            Err(error) => {
                tracing::warn!(%error, "live query cap unreadable; running without it");
                return self.inner.run(ctx, request).await;
            }
        }
        let result = self.inner.run(ctx, request).await;
        if let Err(error) = self.admission.leave(&holder).await {
            // The slot falls out of the set when its lease expires.
            tracing::warn!(%error, "live query slot was not given back");
        }
        result
    }
}
