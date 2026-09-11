//! A concurrency limit in front of a model provider.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Semaphore;

use crate::core::traits::model::ModelProvider;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::model::{ModelError, ModelRequest, ModelResponse};

/// Admits slots model calls at once and queues the rest for up to max_wait.
///
/// slots mirrors llama-server --parallel. The queue stays inside the engine, under the deadline
/// and cancellation of the request.
pub struct Limited {
    inner: Arc<dyn ModelProvider>,
    permits: Semaphore,
    max_wait: Duration,
}

impl Limited {
    /// Wraps a provider. slots must be at least 1. wiring skips the wrapper when unlimited.
    pub fn new(inner: Arc<dyn ModelProvider>, slots: usize, max_wait: Duration) -> Self {
        Self {
            inner,
            permits: Semaphore::new(slots),
            max_wait,
        }
    }
}

#[async_trait]
impl ModelProvider for Limited {
    async fn generate(
        &self,
        ctx: &RequestContext,
        req: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        let wait = self.max_wait.min(ctx.remaining());
        let permit = tokio::select! {
            () = ctx.cancel.cancelled() => return Err(ModelError::Cancelled),
            acquired = tokio::time::timeout(wait, self.permits.acquire()) => acquired,
        };
        let _permit = match permit {
            Ok(Ok(permit)) => permit,
            Ok(Err(e)) => return Err(ModelError::Transport(format!("model limiter closed: {e}"))),
            Err(_) => return Err(ModelError::Busy),
        };
        self.inner.generate(ctx, req).await
    }
}
