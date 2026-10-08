//! Pending actions on the platform agents routes.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Method;
use serde_json::json;
use uuid::Uuid;

use super::client::{Call, PlatformClient, PlatformError, field};
use crate::core::traits::safety::confirmation::ConfirmationStore;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::safety::policy::PendingAction;
use crate::core::types::store::StoreError;

/// Maps a platform error to a StoreError.
#[allow(clippy::needless_pass_by_value)]
fn store(e: PlatformError) -> StoreError {
    StoreError::Database(e.to_string())
}

/// Actions waiting on approval, held by the platform agents module.
pub struct PlatformConfirmations {
    client: PlatformClient,
}

impl PlatformConfirmations {
    /// Wraps a client.
    pub fn new(client: PlatformClient) -> Self {
        Self { client }
    }
}

#[async_trait]
impl ConfirmationStore for PlatformConfirmations {
    async fn hold(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        pending: &PendingAction,
        payload_hash: &str,
        ttl: Duration,
    ) -> Result<(), StoreError> {
        let path = PlatformClient::member(&ctx.user_id, &format!("/pending/{token}"));
        let body = json!({
            "action": pending,
            "payload_hash": payload_hash,
            "ttl_seconds": ttl.as_secs(),
        });
        self.client
            .call(&Call::new(Method::PUT, &path).body(&body))
            .await
            .map_err(store)?;
        Ok(())
    }

    async fn claim(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        approved: bool,
    ) -> Result<Option<PendingAction>, StoreError> {
        let path = PlatformClient::member(&ctx.user_id, &format!("/pending/{token}/claim"));
        let body = json!({ "approved": approved });
        match self
            .client
            .call(&Call::new(Method::POST, &path).body(&body))
            .await
        {
            Ok(value) => field(&value, "action").map(Some).map_err(store),
            Err(e) if e.status() == Some(404) => Ok(None),
            Err(e) => Err(store(e)),
        }
    }
}
