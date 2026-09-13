//! A concurrency limit in front of a model provider.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::core::traits::model::ModelProvider;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::model::{ModelDelta, ModelError, ModelRequest, ModelResponse};

/// Admits slots model calls at once and queues the rest for up to max_wait.
pub struct Limited {
    inner: Arc<dyn ModelProvider>,
    permits: Arc<Semaphore>,
    max_wait: Duration,
}

impl Limited {
    /// Wraps a provider. slots must be at least 1.
    pub fn new(inner: Arc<dyn ModelProvider>, slots: usize, max_wait: Duration) -> Self {
        Self {
            inner,
            permits: Arc::new(Semaphore::new(slots)),
            max_wait,
        }
    }

    /// Waits for a slot within the queue wait and the request budget.
    async fn admit(&self, ctx: &RequestContext) -> Result<OwnedSemaphorePermit, ModelError> {
        let wait = self.max_wait.min(ctx.remaining());
        let permit = tokio::select! {
            () = ctx.cancel.cancelled() => return Err(ModelError::Cancelled),
            acquired = tokio::time::timeout(wait, Arc::clone(&self.permits).acquire_owned()) => acquired,
        };
        match permit {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(e)) => Err(ModelError::Transport(format!("model limiter closed: {e}"))),
            Err(_) => Err(ModelError::Busy),
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
        let _permit = self.admit(ctx).await?;
        self.inner.generate(ctx, req).await
    }

    async fn stream(
        &self,
        ctx: &RequestContext,
        req: ModelRequest,
        deltas: UnboundedSender<ModelDelta>,
    ) -> Result<ModelResponse, ModelError> {
        let _permit = self.admit(ctx).await?;
        self.inner.stream(ctx, req, deltas).await
    }
}
