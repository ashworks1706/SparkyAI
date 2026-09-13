//! Guardrail trait.

use async_trait::async_trait;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::safety::guardrail::{Stage, Verdict};

/// The gate every model response passes, on the capability branch and the answer branch.
/// Decides whether a response proceeds.
#[async_trait]
pub trait Guardrail: Send + Sync {
    /// Checks one response.
    async fn check(&self, ctx: &RequestContext, stage: Stage, text: &str) -> Verdict;
}
