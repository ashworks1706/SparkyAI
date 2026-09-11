//! Conversations and messages tables.

use async_trait::async_trait;
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

use crate::core::traits::conversation::ConversationStore;
use crate::core::types::context::RequestContext;
use crate::core::types::message::Message;
use crate::core::types::store::StoreError;
use crate::stores::postgres::{db, row_limit};

/// Conversations and messages tables.
pub struct PgConversations {
    pool: PgPool,
}

impl PgConversations {
    /// Wraps a pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    async fn user_row_id(&self, ctx: &RequestContext) -> Result<Uuid, StoreError> {
        let row = sqlx::query(
            "insert into users (tenant_id, discord_id, roles) values ($1, $2, $3)
             on conflict (tenant_id, discord_id) do update set roles = excluded.roles
             returning id",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(&ctx.roles)
        .fetch_one(&self.pool)
        .await
        .map_err(db)?;
        row.try_get("id").map_err(db)
    }
}

#[async_trait]
impl ConversationStore for PgConversations {
    async fn ensure(&self, ctx: &RequestContext, channel_id: &str) -> Result<(), StoreError> {
        let user_id = self.user_row_id(ctx).await?;
        sqlx::query(
            "insert into conversations (id, tenant_id, user_id, channel_id) values ($1, $2, $3, $4)
             on conflict (id) do nothing",
        )
        .bind(ctx.conversation_id)
        .bind(&ctx.tenant_id)
        .bind(user_id)
        .bind(channel_id)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn load(&self, ctx: &RequestContext, limit: usize) -> Result<Vec<Message>, StoreError> {
        // A summary stands in for everything before it. Loading past one would carry the turns
        // it replaced as well as the turn that replaced them.
        let rows = sqlx::query(
            "select m.content from messages m join conversations c on c.id = m.conversation_id
             where m.conversation_id = $1 and c.tenant_id = $2
               and m.created_at >= coalesce(
                 (select max(s.created_at) from messages s
                  where s.conversation_id = $1 and s.role = 'summary'),
                 m.created_at)
             order by m.created_at desc limit $3",
        )
        .bind(ctx.conversation_id)
        .bind(&ctx.tenant_id)
        .bind(row_limit(limit))
        .fetch_all(&self.pool)
        .await
        .map_err(db)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows.iter().rev() {
            let value: serde_json::Value = row.try_get("content").map_err(db)?;
            let message = serde_json::from_value::<Message>(value)
                .map_err(|e| StoreError::Database(format!("stored message unreadable: {e}")))?;
            out.push(message);
        }
        Ok(out)
    }

    async fn append(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await.map_err(db)?;
        for m in turns {
            let content =
                serde_json::to_value(m).map_err(|e| StoreError::Database(e.to_string()))?;
            sqlx::query(
                "insert into messages (conversation_id, role, content) values ($1, $2, $3)",
            )
            .bind(ctx.conversation_id)
            .bind(role_str(m))
            .bind(content)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        }
        sqlx::query("update conversations set updated_at = now() where id = $1")
            .bind(ctx.conversation_id)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        tx.commit().await.map_err(db)
    }
}

fn role_str(m: &Message) -> &'static str {
    match m.role {
        crate::core::types::message::Role::System => "system",
        crate::core::types::message::Role::User => "user",
        crate::core::types::message::Role::Assistant => "assistant",
        crate::core::types::message::Role::Tool => "tool",
        crate::core::types::message::Role::Summary => "summary",
    }
}
