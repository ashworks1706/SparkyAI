//! Confirmations table: actions waiting on caller approval.

use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

use crate::core::traits::safety::confirmation::ConfirmationStore;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::safety::policy::PendingAction;
use crate::core::types::store::StoreError;
use crate::stores::postgres::db;

/// Actions waiting on caller approval, in confirmations.
pub struct PgConfirmations {
    pool: PgPool,
}

impl PgConfirmations {
    /// Wraps a pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl ConfirmationStore for PgConfirmations {
    async fn hold(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        pending: &PendingAction,
        payload_hash: &str,
        ttl: Duration,
    ) -> Result<(), StoreError> {
        let action =
            serde_json::to_value(pending).map_err(|e| StoreError::Database(e.to_string()))?;
        let expires_at = Utc::now()
            + chrono::Duration::from_std(ttl)
                .map_err(|e| StoreError::Database(format!("confirmation ttl: {e}")))?;
        sqlx::query(
            "insert into confirmations
               (id, request_id, user_id, action, payload_hash, status, expires_at)
             select $1, $2, u.id, $3, $4, 'pending', $5
             from users u where u.tenant_id = $6 and u.discord_id = $7",
        )
        .bind(token)
        .bind(ctx.request_id)
        .bind(&action)
        .bind(payload_hash)
        .bind(expires_at)
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn claim(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        approved: bool,
    ) -> Result<Option<PendingAction>, StoreError> {
        // One statement. The row moves out of pending as it is read, and the user join limits
        // it to the caller who was asked.
        let row = sqlx::query(
            "update confirmations c
                set status = case when $1 then 'confirmed' else 'denied' end,
                    resolved_at = now()
              from users u
             where c.id = $2
               and c.user_id = u.id
               and u.tenant_id = $3
               and u.discord_id = $4
               and c.status = 'pending'
               and c.expires_at > now()
            returning c.action",
        )
        .bind(approved)
        .bind(token)
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(db)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let action: serde_json::Value = row.try_get("action").map_err(db)?;
        serde_json::from_value(action)
            .map(Some)
            .map_err(|e| StoreError::Database(format!("held action unreadable: {e}")))
    }
}
