//! Guardrail trait.

use async_trait::async_trait;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::safety::guardrail::{Stage, Verdict};

/// The gate every model response passes, on the capability branch and the answer branch.
///
/// Policy classifies a typed action by risk. The guardrail decides whether a response may
/// proceed at all.
#[async_trait]
pub trait Guardrail: Send + Sync {
    /// Checks one response.
    async fn check(&self, ctx: &RequestContext, stage: Stage, text: &str) -> Verdict;
}
