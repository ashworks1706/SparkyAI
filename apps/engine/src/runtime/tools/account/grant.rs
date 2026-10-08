//! Credentials: the per-user token a tool uses, from the grant store, then the shared fallback.

use std::sync::Arc;

use secrecy::{ExposeSecret, SecretString};

use crate::core::traits::oauth::OAuthStore;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::ToolError;
use crate::core::types::tools::oauth::USER_SCOPE;
use crate::runtime::tools::account::oauth::WebOAuthClient;

/// Resolves a provider token for a caller: their own per-user grant, else the shared fallback.
pub struct Credentials {
    store: Arc<dyn OAuthStore>,
    oauth: Option<Arc<WebOAuthClient>>,
    provider: &'static str,
    fallback: SecretString,
}

impl Credentials {
    /// Builds the resolver over the store, the client used to refresh, the provider, and a token.
    pub fn new(
        store: Arc<dyn OAuthStore>,
        oauth: Option<Arc<WebOAuthClient>>,
        provider: &'static str,
        fallback: SecretString,
    ) -> Self {
        Self {
            store,
            oauth,
            provider,
            fallback,
        }
    }

    /// The token for the caller, refreshing an expired grant when it can. None is not connected.
    pub async fn resolve(&self, ctx: &RequestContext) -> Result<Option<SecretString>, ToolError> {
        let grant = self
            .store
            .load_grant(USER_SCOPE, &ctx.user_id, self.provider)
            .await
            .map_err(|error| {
                tracing::error!(%error, provider = self.provider, "could not read a grant");
                ToolError::Failed("could not read your connection".to_owned())
            })?;
        if let Some(mut grant) = grant {
            if grant.expired() {
                let Some(oauth) = &self.oauth else {
                    return Ok(None);
                };
                match oauth.refresh(&grant).await {
                    Ok(fresh) => {
                        if let Err(error) = self
                            .store
                            .save_grant(USER_SCOPE, &ctx.user_id, self.provider, &fresh)
                            .await
                        {
                            tracing::warn!(%error, "could not save a refreshed grant");
                        }
                        grant = fresh;
                    }
                    Err(error) => {
                        tracing::info!(%error, provider = self.provider, "grant not refreshed");
                        return Ok(None);
                    }
                }
            }
            return Ok(Some(grant.access_token));
        }
        if self.fallback.expose_secret().trim().is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.fallback.clone()))
        }
    }
}
