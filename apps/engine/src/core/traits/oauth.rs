//! OAuthStore trait: per-user grants, the consent state of a login, or a login another service runs.

use std::time::Duration;

use async_trait::async_trait;

use crate::core::types::store::StoreError;
use crate::core::types::tools::oauth::{Consent, OAuthTokens};

/// Holds each caller's per-provider grant, and the pending state of a login in progress.
#[async_trait]
pub trait OAuthStore: Send + Sync {
    /// Saves or replaces the caller's grant for a provider.
    async fn save_grant(
        &self,
        tenant: &str,
        user: &str,
        provider: &str,
        tokens: &OAuthTokens,
    ) -> Result<(), StoreError>;

    /// The caller's grant for a provider, if one is stored.
    async fn load_grant(
        &self,
        tenant: &str,
        user: &str,
        provider: &str,
    ) -> Result<Option<OAuthTokens>, StoreError>;

    /// Removes the caller's grant for a provider. Returns whether one was there.
    async fn delete_grant(
        &self,
        tenant: &str,
        user: &str,
        provider: &str,
    ) -> Result<bool, StoreError>;

    /// Records a pending consent under state until ttl passes.
    async fn begin_consent(
        &self,
        state: &str,
        tenant: &str,
        user: &str,
        provider: &str,
        ttl: Duration,
    ) -> Result<(), StoreError>;

    /// Claims a pending consent by state; one use, gone once taken or expired.
    async fn take_consent(&self, state: &str) -> Result<Option<Consent>, StoreError>;

    /// Whether the store runs the provider login itself, so login_link replaces the consent routes.
    fn hosts_login(&self) -> bool {
        false
    }

    /// The link that starts a login the store runs, or None when it offers no login for provider.
    async fn login_link(&self, _user: &str, _provider: &str) -> Result<Option<String>, StoreError> {
        Ok(None)
    }
}
