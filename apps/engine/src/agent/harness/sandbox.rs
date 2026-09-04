//! `Sandbox` trait: browser tasks in an isolated worker. Phase 7.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::sandbox::{BrowserResult, BrowserTask, SandboxError};

/// Runs browser tasks outside the engine process.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// Runs one task in the user's isolated session.
    async fn run(
        &self,
        ctx: &RequestContext,
        task: &BrowserTask,
    ) -> Result<BrowserResult, SandboxError>;
}
