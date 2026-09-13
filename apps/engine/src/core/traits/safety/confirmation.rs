//! ConfirmationStore trait.

use std::time::Duration;

use async_trait::async_trait;
use uuid::Uuid;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::safety::policy::PendingAction;
use crate::core::types::store::StoreError;

/// Holds actions waiting on caller approval.
#[async_trait]
pub trait ConfirmationStore: Send + Sync {
    /// Holds pending under token until the caller answers or ttl passes.
    async fn hold(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        pending: &PendingAction,
        payload_hash: &str,
        ttl: Duration,
    ) -> Result<(), StoreError>;

    /// Answers a held confirmation: what to run, or None if unowned, answered, expired; one use.
    async fn claim(
        &self,
        ctx: &RequestContext,
        token: Uuid,
        approved: bool,
    ) -> Result<Option<PendingAction>, StoreError>;
}
