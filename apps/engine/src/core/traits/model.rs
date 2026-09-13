//! ModelProvider trait.

use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::model::{ModelDelta, ModelError, ModelRequest, ModelResponse};

/// A chat model behind an OpenAI-compatible endpoint, or a test double.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Runs one completion within the deadline and cancellation of the context.
    async fn generate(
        &self,
        ctx: &RequestContext,
        req: ModelRequest,
    ) -> Result<ModelResponse, ModelError>;

    /// Runs one completion like generate, streaming pieces to deltas; non-streaming sends nothing.
    async fn stream(
        &self,
        ctx: &RequestContext,
        req: ModelRequest,
        deltas: UnboundedSender<ModelDelta>,
    ) -> Result<ModelResponse, ModelError> {
        drop(deltas);
        self.generate(ctx, req).await
    }
}
