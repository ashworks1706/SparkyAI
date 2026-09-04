//! `ConversationStore` trait.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::message::Message;
use crate::core::types::store::StoreError;

/// Durable conversation history, scoped by tenant and conversation.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// Ensures the conversation row exists for this request's user.
    async fn ensure(&self, ctx: &RequestContext, channel_id: &str) -> Result<(), StoreError>;
    /// Loads the most recent `limit` turns, oldest first.
    async fn load(&self, ctx: &RequestContext, limit: usize) -> Result<Vec<Message>, StoreError>;
    /// Appends turns in order.
    async fn append(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError>;
}
