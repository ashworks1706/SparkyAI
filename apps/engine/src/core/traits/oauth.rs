//! OAuthStore trait: the caller's linked accounts, held and logged in by another service.

use async_trait::async_trait;

use crate::core::types::store::StoreError;
use crate::core::types::tools::oauth::OAuthTokens;

/// Each caller's per-provider grant, and the login that links one. The store runs the login.
#[async_trait]
pub trait OAuthStore: Send + Sync {
    /// The caller's current grant for a provider, if one is linked.
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

    /// The link that starts the provider login, or None when the store offers no login for it.
    async fn login_link(&self, user: &str, provider: &str) -> Result<Option<String>, StoreError>;
}
