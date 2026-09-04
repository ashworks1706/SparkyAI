//! `Policy` trait.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::policy::{Decision, PolicyError, ProposedAction};

/// Decides whether a proposed action runs.
#[async_trait]
pub trait Policy: Send + Sync {
    /// Evaluates one action for one request.
    async fn authorize(
        &self,
        ctx: &RequestContext,
        action: &ProposedAction,
    ) -> Result<Decision, PolicyError>;
}
