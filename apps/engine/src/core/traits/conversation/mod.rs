//! ConversationStore trait.

pub mod compaction;

use async_trait::async_trait;
use uuid::Uuid;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Stored;
use crate::core::types::conversation::message::Message;
use crate::core::types::store::StoreError;

/// Durable conversation history, scoped by tenant, user, and conversation.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// Ensures the conversation row exists for the user of this request, created at the
    /// visibility of the request.
    ///
    /// # Errors
    /// [StoreError::NotOwned] when the id belongs to another user or tenant.
    async fn ensure(&self, ctx: &RequestContext, channel_id: &str) -> Result<(), StoreError>;
    /// Whether the conversation exists and belongs to the user of this request.
    async fn owns(&self, ctx: &RequestContext) -> Result<bool, StoreError>;
    /// Loads the history of a conversation the caller owns, oldest first: the newest summary,
    /// then at most limit of the messages after the last message it covers.
    async fn load(&self, ctx: &RequestContext, limit: usize) -> Result<Vec<Stored>, StoreError>;
    /// Appends turns in order.
    async fn append(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError>;
    /// Stores a summary that stands in for every message up to and including position covers.
    async fn append_summary(
        &self,
        ctx: &RequestContext,
        summary: &Message,
        covers: i64,
    ) -> Result<(), StoreError>;
    /// The most recently updated open conversation of the caller in channel_id at the
    /// visibility of the request.
    async fn latest(
        &self,
        ctx: &RequestContext,
        channel_id: &str,
    ) -> Result<Option<Uuid>, StoreError>;
    /// Ends every open conversation of the caller in channel_id, at any visibility. Returns
    /// how many ended.
    async fn end(&self, ctx: &RequestContext, channel_id: &str) -> Result<u64, StoreError>;
}
