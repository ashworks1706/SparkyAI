//! `ConfirmationStore` trait.

use std::time::Duration;

use async_trait::async_trait;
use uuid::Uuid;

use crate::core::types::context::RequestContext;
use crate::core::types::policy::PendingAction;
use crate::core::types::store::StoreError;

/// Holds actions waiting on their caller's approval.
#[async_trait]
pub trait ConfirmationStore: Send + Sync {
    /// Holds `pending` under `token` until the caller answers or `ttl` passes.
    async fn hold(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        pending: &PendingAction,
        payload_hash: &str,
        ttl: Duration,
    ) -> Result<(), StoreError>;

    /// Answers a held confirmation and returns what to run, or `None` when the token is not
    /// this caller's, has already been answered, or has expired.
    ///
    /// Implementations resolve in one statement so a token cannot be claimed twice.
    async fn claim(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        approved: bool,
    ) -> Result<Option<PendingAction>, StoreError>;
}
