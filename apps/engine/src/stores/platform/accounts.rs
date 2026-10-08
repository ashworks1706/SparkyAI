//! Linked accounts on the platform accounts module. The platform runs the login and refreshes tokens.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Method;
use secrecy::SecretString;
use serde_json::Value;

use super::client::{Call, PlatformClient, PlatformError, field, optional_timestamp, segment};
use crate::core::traits::oauth::OAuthStore;
use crate::core::types::store::StoreError;
use crate::core::types::tools::oauth::{Consent, OAuthTokens};

/// Maps a platform error to a StoreError.
#[allow(clippy::needless_pass_by_value)]
fn store(e: PlatformError) -> StoreError {
    StoreError::Database(e.to_string())
}

/// Per-member grants held by the platform. Scope is the token organization; tenant is not sent.
pub struct PlatformAccounts {
    client: PlatformClient,
}

impl PlatformAccounts {
    /// Wraps a client.
    pub fn new(client: PlatformClient) -> Self {
        Self { client }
    }
}

/// The access token of a token response. No refresh token leaves the platform.
fn tokens(value: &Value) -> Result<OAuthTokens, PlatformError> {
    let access: String = field(value, "access_token")?;
    Ok(OAuthTokens {
        access_token: SecretString::from(access),
        refresh_token: None,
        scopes: field(value, "scopes")?,
        expires_at: optional_timestamp(value, "expires_at")?,
    })
}

#[async_trait]
impl OAuthStore for PlatformAccounts {
    async fn save_grant(
        &self,
        _tenant: &str,
        _user: &str,
        provider: &str,
        _tokens: &OAuthTokens,
    ) -> Result<(), StoreError> {
        Err(StoreError::Database(format!(
            "the platform saves {provider} grants from its own login"
        )))
    }

    async fn load_grant(
        &self,
        _tenant: &str,
        user: &str,
        provider: &str,
    ) -> Result<Option<OAuthTokens>, StoreError> {
        let path = PlatformClient::account(user, &format!("/{}/token", segment(provider)));
        match self.client.call(&Call::new(Method::GET, &path)).await {
            Ok(value) => tokens(&value).map(Some).map_err(store),
            Err(e) if matches!(e.status(), Some(404 | 409)) => {
                tracing::info!(provider, error = %e, "no usable platform grant");
                Ok(None)
            }
            Err(e) => Err(store(e)),
        }
    }

    async fn delete_grant(
        &self,
        _tenant: &str,
        user: &str,
        provider: &str,
    ) -> Result<bool, StoreError> {
        let path = PlatformClient::account(user, &format!("/{}", segment(provider)));
        let value = self
            .client
            .call(&Call::new(Method::DELETE, &path))
            .await
            .map_err(store)?;
        field(&value, "removed").map_err(store)
    }

    async fn begin_consent(
        &self,
        _state: &str,
        _tenant: &str,
        _user: &str,
        _provider: &str,
        _ttl: Duration,
    ) -> Result<(), StoreError> {
        Err(StoreError::Database(
            "the platform runs the login; ask it for a link".into(),
        ))
    }

    async fn take_consent(&self, _state: &str) -> Result<Option<Consent>, StoreError> {
        Ok(None)
    }

    fn hosts_login(&self) -> bool {
        true
    }

    async fn login_link(&self, user: &str, provider: &str) -> Result<Option<String>, StoreError> {
        let path = PlatformClient::account(user, &format!("/{}/login", segment(provider)));
        match self.client.call(&Call::new(Method::POST, &path)).await {
            Ok(value) => field(&value, "url").map(Some).map_err(store),
            Err(e) if e.status() == Some(404) => Ok(None),
            Err(e) => Err(store(e)),
        }
    }
}
