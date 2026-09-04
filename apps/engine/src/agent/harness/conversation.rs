//! `ConversationStore` trait: load and append turns.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::message::Message;

/// Store failures shared by conversation and memory stores.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The database rejected or could not run the operation.
    #[error("store: {0}")]
    Database(String),
    /// A row the request needs does not exist.
    #[error("not found: {0}")]
    NotFound(String),
}

/// Durable conversation history, scoped by tenant and conversation.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// Ensures the conversation row exists for this request's user and returns its id.
    async fn ensure(&self, ctx: &RequestContext, channel_id: &str) -> Result<(), StoreError>;
    /// Loads the most recent `limit` turns, oldest first.
    async fn load(&self, ctx: &RequestContext, limit: usize) -> Result<Vec<Message>, StoreError>;
    /// Appends turns in order.
    async fn append(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError>;
}
