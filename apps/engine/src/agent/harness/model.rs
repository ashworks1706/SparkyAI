//! `ModelProvider` trait.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::model::{ModelError, ModelRequest, ModelResponse};

/// A chat model behind an OpenAI-compatible endpoint, or a test double.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Runs one completion within the context's deadline and cancellation.
    async fn generate(
        &self,
        ctx: &RequestContext,
        req: ModelRequest,
    ) -> Result<ModelResponse, ModelError>;
}
