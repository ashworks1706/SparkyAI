//! Sandbox trait.

use async_trait::async_trait;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::sandbox::{SandboxError, SandboxOutput, SandboxRequest};

/// Runs a command in an isolated environment.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// Runs one command and returns what it produced.
    async fn run(
        &self,
        ctx: &RequestContext,
        request: &SandboxRequest,
    ) -> Result<SandboxOutput, SandboxError>;
}
