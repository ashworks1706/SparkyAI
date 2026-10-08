//! Conversations and their messages on the platform agents routes.

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Value, json};
use uuid::Uuid;

use super::client::{Call, PlatformClient, PlatformError, field, segment};
use crate::core::traits::conversation::ConversationStore;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Stored;
use crate::core::types::conversation::message::{Message, Role};
use crate::core::types::store::StoreError;

/// Maps a platform error to a StoreError; 409 means another member holds the conversation.
#[allow(clippy::needless_pass_by_value)]
fn store(e: PlatformError) -> StoreError {
    match e.status() {
        Some(409) => StoreError::NotOwned,
        _ => StoreError::Database(e.to_string()),
    }
}

/// The platform row role of a message. The whole message, its own role included, is the content.
fn row_role(message: &Message) -> &'static str {
    match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool | Role::Summary => "tool",
    }
}

/// Conversations held by the platform agents module.
pub struct PlatformConversations {
    client: PlatformClient,
}

impl PlatformConversations {
    /// Wraps a client.
    pub fn new(client: PlatformClient) -> Self {
        Self { client }
    }

    /// The path of the request conversation followed by rest.
    fn conversation(ctx: &RequestContext, rest: &str) -> String {
        PlatformClient::member(
            &ctx.user_id,
            &format!("/conversations/{}{rest}", ctx.conversation_id),
        )
    }

    /// The path of one channel of the caller followed by rest.
    fn channel(ctx: &RequestContext, channel_id: &str, rest: &str) -> String {
        PlatformClient::member(
            &ctx.user_id,
            &format!("/channels/{}{rest}", segment(channel_id)),
        )
    }
}

/// One message as JSON, for the content column.
fn content(message: &Message) -> Result<Value, StoreError> {
    serde_json::to_value(message).map_err(|e| StoreError::Database(e.to_string()))
}

#[async_trait]
impl ConversationStore for PlatformConversations {
    async fn ensure(&self, ctx: &RequestContext, channel_id: &str) -> Result<(), StoreError> {
        let body = json!({"channel_id": channel_id, "visibility": ctx.visibility.as_str()});
        let path = Self::conversation(ctx, "");
        self.client
            .call(&Call::new(Method::PUT, &path).body(&body))
            .await
            .map_err(store)?;
        Ok(())
    }

    async fn owns(&self, ctx: &RequestContext) -> Result<bool, StoreError> {
        let query = [("visibility", ctx.visibility.as_str().to_owned())];
        let path = Self::conversation(ctx, "");
        let value = self
            .client
            .call(&Call::new(Method::GET, &path).query(&query))
            .await
            .map_err(store)?;
        field(&value, "owned").map_err(store)
    }

    async fn load(&self, ctx: &RequestContext, limit: usize) -> Result<Vec<Stored>, StoreError> {
        let query = [("limit", limit.max(1).to_string())];
        let path = Self::conversation(ctx, "/messages");
        let value = self
            .client
            .call(&Call::new(Method::GET, &path).query(&query))
            .await
            .map_err(store)?;
        let rows: Vec<Value> = field(&value, "messages").map_err(store)?;
        rows.iter()
            .map(|row| {
                let message: Message = field(row, "content").map_err(store)?;
                Ok(Stored {
                    position: field(row, "position").map_err(store)?,
                    message,
                })
            })
            .collect()
    }

    async fn append(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError> {
        if turns.is_empty() {
            return Ok(());
        }
        let messages = turns
            .iter()
            .map(|m| Ok(json!({"role": row_role(m), "content": content(m)?})))
            .collect::<Result<Vec<Value>, StoreError>>()?;
        let body = json!({ "messages": messages });
        let path = Self::conversation(ctx, "/messages");
        self.client
            .call(&Call::new(Method::POST, &path).body(&body))
            .await
            .map_err(store)?;
        Ok(())
    }

    async fn append_summary(
        &self,
        ctx: &RequestContext,
        summary: &Message,
        covers: i64,
    ) -> Result<(), StoreError> {
        let body = json!({"content": content(summary)?, "covers": covers});
        let path = Self::conversation(ctx, "/summary");
        self.client
            .call(&Call::new(Method::POST, &path).body(&body))
            .await
            .map_err(store)?;
        Ok(())
    }

    async fn latest(
        &self,
        ctx: &RequestContext,
        channel_id: &str,
    ) -> Result<Option<Uuid>, StoreError> {
        let query = [("visibility", ctx.visibility.as_str().to_owned())];
        let path = Self::channel(ctx, channel_id, "/latest");
        let value = self
            .client
            .call(&Call::new(Method::GET, &path).query(&query))
            .await
            .map_err(store)?;
        field(&value, "id").map_err(store)
    }

    async fn end(&self, ctx: &RequestContext, channel_id: &str) -> Result<u64, StoreError> {
        let path = Self::channel(ctx, channel_id, "/end");
        let value = self
            .client
            .call(&Call::new(Method::POST, &path))
            .await
            .map_err(store)?;
        field(&value, "ended").map_err(store)
    }
}
