//! Grant store doubles: one that runs the login itself, as the platform does.

use std::time::Duration;

use async_trait::async_trait;

use crate::core::traits::oauth::OAuthStore;
use crate::core::types::store::StoreError;
use crate::core::types::tools::oauth::{Consent, OAuthTokens};

/// Hands out a login link for canvas only and holds no grants; every other call is refused.
pub struct HostedLogins;

#[async_trait]
impl OAuthStore for HostedLogins {
    async fn save_grant(
        &self,
        _t: &str,
        _u: &str,
        _p: &str,
        _tk: &OAuthTokens,
    ) -> Result<(), StoreError> {
        Err(StoreError::Database("hosted".into()))
    }

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

    async fn begin_consent(
        &self,
        _s: &str,
        _t: &str,
        _u: &str,
        _p: &str,
        _ttl: Duration,
    ) -> Result<(), StoreError> {
        Err(StoreError::Database("hosted".into()))
    }

    async fn take_consent(&self, _state: &str) -> Result<Option<Consent>, StoreError> {
        Ok(None)
    }

    fn hosts_login(&self) -> bool {
        true
    }

    async fn login_link(&self, user: &str, provider: &str) -> Result<Option<String>, StoreError> {
        Ok((provider == "canvas").then(|| format!("https://platform.test/start/{user}")))
    }
}
