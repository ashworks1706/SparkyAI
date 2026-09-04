//! `Tool` trait and the `ToolSet` registry.

use std::collections::BTreeMap;
use std::sync::Arc;

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

/// The tools available to one agent, keyed by name.
#[derive(Default, Clone)]
pub struct ToolSet {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a tool, replacing any with the same name.
    pub fn with(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.insert(tool.definition().name, tool);
        self
    }

    /// Looks a tool up by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Definitions in name order, for the model.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Whether any tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl std::fmt::Debug for ToolSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.tools.keys()).finish()
    }
}
