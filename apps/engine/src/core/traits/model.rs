//! ModelProvider trait.

use async_trait::async_trait;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::model::{ModelError, ModelRequest, ModelResponse};

/// A chat model behind an OpenAI-compatible endpoint, or a test double.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Runs one completion within the deadline and cancellation of the context.
    async fn generate(
        &self,
        ctx: &RequestContext,
        req: ModelRequest,
    ) -> Result<ModelResponse, ModelError>;
}
