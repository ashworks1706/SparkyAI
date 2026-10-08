//! Probe trait: one dependency readiness asks about.

use async_trait::async_trait;

/// A dependency the engine needs to answer, checked on each readiness request.
#[async_trait]
pub trait Probe: Send + Sync {
    /// The name the readiness report gives it.
    fn name(&self) -> &'static str;
    /// Whether it answered.
    async fn ready(&self) -> bool;
}
