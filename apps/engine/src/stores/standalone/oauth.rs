//! oauth_grants and oauth_states: per-user grants and the pending state of a login.

use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use secrecy::{ExposeSecret, SecretString};
use sqlx::Row;
use sqlx::postgres::PgPool;

use crate::core::traits::oauth::OAuthStore;
use crate::core::types::store::StoreError;
use crate::core::types::tools::oauth::{Consent, OAuthTokens};
use crate::stores::standalone::postgres::db;

/// Per-user OAuth grants and login states, in oauth_grants and oauth_states.
pub struct PgOAuth {
    pool: PgPool,
}

impl PgOAuth {
    /// Wraps a pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl OAuthStore for PgOAuth {
    async fn save_grant(
        &self,
        tenant: &str,
        user: &str,
        provider: &str,
        tokens: &OAuthTokens,
    ) -> Result<(), StoreError> {
        let refresh = tokens
            .refresh_token
            .as_ref()
            .map(|s| s.expose_secret().to_owned());
        sqlx::query(
            "insert into oauth_grants
               (tenant_id, user_id, provider, access_token, refresh_token, scopes, expires_at,
                updated_at)
             values ($1, $2, $3, $4, $5, $6, $7, now())
             on conflict (tenant_id, user_id, provider) do update set
               access_token  = excluded.access_token,
               refresh_token = coalesce(excluded.refresh_token, oauth_grants.refresh_token),
               scopes        = excluded.scopes,
               expires_at    = excluded.expires_at,
               updated_at    = now()",
        )
        .bind(tenant)
        .bind(user)
        .bind(provider)
        .bind(tokens.access_token.expose_secret())
        .bind(refresh)
        .bind(&tokens.scopes)
        .bind(tokens.expires_at)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn load_grant(
        &self,
        tenant: &str,
        user: &str,
        provider: &str,
    ) -> Result<Option<OAuthTokens>, StoreError> {
        let row = sqlx::query(
            "select access_token, refresh_token, scopes, expires_at from oauth_grants
             where tenant_id = $1 and user_id = $2 and provider = $3",
        )
        .bind(tenant)
        .bind(user)
        .bind(provider)
        .fetch_optional(&self.pool)
        .await
        .map_err(db)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let access: String = row.try_get("access_token").map_err(db)?;
        let refresh: Option<String> = row.try_get("refresh_token").map_err(db)?;
        let scopes: Vec<String> = row.try_get("scopes").map_err(db)?;
        let expires_at: Option<DateTime<Utc>> = row.try_get("expires_at").map_err(db)?;
        Ok(Some(OAuthTokens {
            access_token: SecretString::from(access),
            refresh_token: refresh.map(SecretString::from),
            scopes,
            expires_at,
        }))
    }

    async fn delete_grant(
        &self,
        tenant: &str,
        user: &str,
        provider: &str,
    ) -> Result<bool, StoreError> {
        let done = sqlx::query(
            "delete from oauth_grants where tenant_id = $1 and user_id = $2 and provider = $3",
        )
        .bind(tenant)
        .bind(user)
        .bind(provider)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(done.rows_affected() > 0)
    }

    async fn begin_consent(
        &self,
        state: &str,
        tenant: &str,
        user: &str,
        provider: &str,
        ttl: Duration,
    ) -> Result<(), StoreError> {
        let expires_at = Utc::now()
            + chrono::Duration::from_std(ttl)
                .map_err(|e| StoreError::Database(format!("consent ttl: {e}")))?;
        sqlx::query(
            "insert into oauth_states (state, tenant_id, user_id, provider, expires_at)
             values ($1, $2, $3, $4, $5)",
        )
        .bind(state)
        .bind(tenant)
        .bind(user)
        .bind(provider)
        .bind(expires_at)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(())
    }

    async fn take_consent(&self, state: &str) -> Result<Option<Consent>, StoreError> {
        let row = sqlx::query(
            "delete from oauth_states where state = $1 and expires_at > now()
             returning tenant_id, user_id, provider",
        )
        .bind(state)
        .fetch_optional(&self.pool)
        .await
        .map_err(db)?;
        let Some(row) = row else {
            return Ok(None);
        };
        Ok(Some(Consent {
            tenant_id: row.try_get("tenant_id").map_err(db)?,
            user_id: row.try_get("user_id").map_err(db)?,
            provider: row.try_get("provider").map_err(db)?,
        }))
    }
}
