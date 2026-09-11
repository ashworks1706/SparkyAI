//! Compactor trait.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::message::Message;
use crate::core::types::model::ModelError;

/// Replaces the turns that would be dropped with one turn standing in for them.
#[async_trait]
pub trait Compactor: Send + Sync {
    /// Compacts turns into a single summary turn.
    ///
    /// # Errors
    /// Returns [`ModelError`] when the call fails or answers with nothing.
    async fn compact(&self, ctx: &RequestContext, turns: &[Message])
    -> Result<Message, ModelError>;
}
