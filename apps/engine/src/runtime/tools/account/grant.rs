//! Credentials: the per-user token a tool uses, from the caller's linked account.

use std::sync::Arc;

use secrecy::SecretString;

use crate::core::traits::oauth::OAuthStore;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::ToolError;
use crate::core::types::tools::oauth::USER_SCOPE;

/// Resolves a provider token for a caller from their linked account.
pub struct Credentials {
    store: Arc<dyn OAuthStore>,
    provider: &'static str,
}

impl Credentials {
    /// Builds the resolver over the grant store for one provider.
    pub fn new(store: Arc<dyn OAuthStore>, provider: &'static str) -> Self {
        Self { store, provider }
    }

    /// The token for the caller. None is not connected, or a grant that has expired.
    pub async fn resolve(&self, ctx: &RequestContext) -> Result<Option<SecretString>, ToolError> {
        let grant = self
            .store
            .load_grant(USER_SCOPE, &ctx.user_id, self.provider)
            .await
            .map_err(|error| {
                tracing::error!(%error, provider = self.provider, "could not read a grant");
                ToolError::Failed("could not read your connection".to_owned())
            })?;
        Ok(grant
            .filter(|grant| !grant.expired())
            .map(|grant| grant.access_token))
    }
}
