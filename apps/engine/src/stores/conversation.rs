//! Conversations and messages tables.

use async_trait::async_trait;
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

use crate::core::traits::conversation::ConversationStore;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Stored;
use crate::core::types::conversation::message::{Message, Role};
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
}

#[async_trait]
impl ConversationStore for PgConversations {
    async fn ensure(&self, ctx: &RequestContext, channel_id: &str) -> Result<(), StoreError> {
        // An id held by another caller, or at another channel or visibility, writes nothing and
        // reads as not owned. The users row is written only when it is new or its roles change.
        let row = sqlx::query(
            "with existing as (
               select c.tenant_id = $2 and u.tenant_id = $2 and u.discord_id = $3
                      and c.channel_id = $4 and c.visibility = $5 as ok
               from conversations c join users u on u.id = c.user_id
               where c.id = $1
             ),
             caller as (
               insert into users (tenant_id, discord_id, roles)
               select $2, $3, $6 where not exists (select 1 from existing)
               on conflict (tenant_id, discord_id) do update set roles = excluded.roles
               where users.roles is distinct from excluded.roles
               returning id
             ),
             owner as (
               select id from caller
               union all
               select id from users where tenant_id = $2 and discord_id = $3
               limit 1
             ),
             created as (
               insert into conversations (id, tenant_id, user_id, channel_id, visibility)
               select $1, $2, owner.id, $4, $5 from owner
               on conflict (id) do nothing
               returning id
             )
             select coalesce((select ok from existing), exists (select 1 from created)) as ok",
        )
        .bind(ctx.conversation_id)
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(channel_id)
        .bind(ctx.visibility.as_str())
        .bind(&ctx.roles)
        .fetch_one(&self.pool)
        .await
        .map_err(db)?;
        let ok: Option<bool> = row.try_get("ok").map_err(db)?;
        if ok == Some(true) {
            Ok(())
        } else {
            Err(StoreError::NotOwned)
        }
    }

    async fn owns(&self, ctx: &RequestContext) -> Result<bool, StoreError> {
        let row = sqlx::query(
            "select exists (
               select 1 from conversations c join users u on u.id = c.user_id
               where c.id = $1 and c.tenant_id = $2 and u.tenant_id = $2 and u.discord_id = $3
                 and c.visibility = $4
             ) as owned",
        )
        .bind(ctx.conversation_id)
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(ctx.visibility.as_str())
        .fetch_one(&self.pool)
        .await
        .map_err(db)?;
        row.try_get("owned").map_err(db)
    }

    async fn load(&self, ctx: &RequestContext, limit: usize) -> Result<Vec<Stored>, StoreError> {
        let rows = sqlx::query(
            "with owned as (
               select c.id from conversations c join users u on u.id = c.user_id
               where c.id = $1 and c.tenant_id = $2 and u.tenant_id = $2 and u.discord_id = $4
             ),
             latest as (
               select coalesce(s.covers_seq, s.seq) as position, s.content from messages s
               where s.conversation_id = (select id from owned) and s.role = 'summary'
               order by s.seq desc limit 1
             ),
             recent as (
               select m.seq as position, m.content from messages m
               where m.conversation_id = (select id from owned) and m.role <> 'summary'
                 and m.seq > coalesce((select position from latest), 0)
               order by m.seq desc limit $3
             )
             select position, content from latest
             union all
             select position, content from recent
             order by position",
        )
        .bind(ctx.conversation_id)
        .bind(&ctx.tenant_id)
        .bind(row_limit(limit))
        .bind(&ctx.user_id)
        .fetch_all(&self.pool)
        .await
        .map_err(db)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let value: serde_json::Value = row.try_get("content").map_err(db)?;
            let message = serde_json::from_value::<Message>(value)
                .map_err(|e| StoreError::Database(format!("stored message unreadable: {e}")))?;
            out.push(Stored {
                position: row.try_get("position").map_err(db)?,
                message,
            });
        }
        Ok(out)
    }

    async fn latest(
        &self,
        ctx: &RequestContext,
        channel_id: &str,
    ) -> Result<Option<Uuid>, StoreError> {
        let row = sqlx::query(
            "select c.id from conversations c join users u on u.id = c.user_id
             where c.tenant_id = $1 and u.tenant_id = $1 and u.discord_id = $2
               and c.channel_id = $3 and c.visibility = $4 and c.ended_at is null
             order by c.updated_at desc limit 1",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(channel_id)
        .bind(ctx.visibility.as_str())
        .fetch_optional(&self.pool)
        .await
        .map_err(db)?;
        row.map(|r| r.try_get("id").map_err(db)).transpose()
    }

    async fn end(&self, ctx: &RequestContext, channel_id: &str) -> Result<u64, StoreError> {
        let done = sqlx::query(
            "update conversations c set ended_at = now()
             from users u
             where u.id = c.user_id and u.tenant_id = $1 and u.discord_id = $2
               and c.tenant_id = $1 and c.channel_id = $3 and c.ended_at is null",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(channel_id)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(done.rows_affected())
    }

    async fn append(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError> {
        self.write(ctx, turns, None).await
    }

    async fn append_summary(
        &self,
        ctx: &RequestContext,
        summary: &Message,
        covers: i64,
    ) -> Result<(), StoreError> {
        self.write(ctx, std::slice::from_ref(summary), Some(covers))
            .await
    }
}

impl PgConversations {
    /// Appends turns in one transaction, each covering covers when it is a summary.
    async fn write(
        &self,
        ctx: &RequestContext,
        turns: &[Message],
        covers: Option<i64>,
    ) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await.map_err(db)?;
        let touched = sqlx::query(
            "update conversations c set updated_at = now() from users u
             where c.id = $1 and c.tenant_id = $2
               and u.id = c.user_id and u.tenant_id = $2 and u.discord_id = $3",
        )
        .bind(ctx.conversation_id)
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        if touched.rows_affected() != 1 {
            return Err(StoreError::NotOwned);
        }
        for m in turns {
            let content =
                serde_json::to_value(m).map_err(|e| StoreError::Database(e.to_string()))?;
            sqlx::query(
                "insert into messages (conversation_id, role, content, covers_seq)
                 values ($1, $2, $3, $4)",
            )
            .bind(ctx.conversation_id)
            .bind(role_str(m))
            .bind(content)
            .bind(covers.filter(|_| m.role == Role::Summary))
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        }
        tx.commit().await.map_err(db)
    }
}

/// The messages.role value of a message.
fn role_str(m: &Message) -> &'static str {
    match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        Role::Summary => "summary",
    }
}
