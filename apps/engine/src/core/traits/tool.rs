//! `Tool` trait.

use async_trait::async_trait;
use serde_json::Value;

use crate::core::types::context::RequestContext;
use crate::core::types::tool::{ToolDefinition, ToolError, ToolOutput};

/// A callable capability.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Name, description, schema, and risk.
    fn definition(&self) -> ToolDefinition;
    /// Executes with validated JSON arguments.
    async fn call(&self, ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError>;
}
