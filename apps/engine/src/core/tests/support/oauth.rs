//! Grant store doubles: one that hands out the platform's login link, one with a linked account.

use async_trait::async_trait;
use secrecy::SecretString;

use crate::core::traits::oauth::OAuthStore;
use crate::core::types::store::StoreError;
use crate::core::types::tools::oauth::OAuthTokens;

/// Hands out a login link for canvas only and holds no grants.
pub struct HostedLogins;

#[async_trait]
impl OAuthStore for HostedLogins {
    async fn load_grant(
        &self,
        _t: &str,
        _u: &str,
        _p: &str,
    ) -> Result<Option<OAuthTokens>, StoreError> {
        Ok(None)
    }

    async fn delete_grant(&self, _t: &str, _u: &str, provider: &str) -> Result<bool, StoreError> {
        Ok(provider == "canvas")
    }

    async fn login_link(&self, user: &str, provider: &str) -> Result<Option<String>, StoreError> {
        Ok((provider == "canvas").then(|| format!("https://platform.test/start/{user}")))
    }
}

/// Holds one linked account for every provider when a token is set, and none otherwise.
pub struct Linked(pub Option<&'static str>);

#[async_trait]
impl OAuthStore for Linked {
    async fn load_grant(
        &self,
        _t: &str,
        _u: &str,
        _p: &str,
    ) -> Result<Option<OAuthTokens>, StoreError> {
        Ok(self.0.map(|token| OAuthTokens {
            access_token: SecretString::from(token),
            expires_at: None,
        }))
    }

    async fn delete_grant(&self, _t: &str, _u: &str, _p: &str) -> Result<bool, StoreError> {
        Ok(self.0.is_some())
    }

    async fn login_link(&self, _user: &str, _provider: &str) -> Result<Option<String>, StoreError> {
        Ok(None)
    }
}
