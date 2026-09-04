//! `Tool` trait, `ToolDefinition`, `RiskClass`, registry.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::types::context::RequestContext;

/// What a tool can do to the world. Drives `Policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskClass {
    /// Reads public data. Runs freely.
    ReadPublic,
    /// Reads inside the user's own authenticated session.
    ReadAuthenticated,
    /// Drafts or fills without submitting.
    PrepareWrite,
    /// Posts, creates, books, submits. Confirmed immediately before.
    ExternalWrite,
    /// Deletes or cancels. Confirmed immediately before.
    Destructive,
    /// Never allowed.
    Forbidden,
}

/// What the model sees, and what `Policy` classifies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique name the model calls.
    pub name: String,
    /// What it does, for the model.
    pub description: String,
    /// JSON Schema for `arguments`.
    pub parameters: Value,
    /// Risk classification.
    pub risk: RiskClass,
}

/// What a tool returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Text handed back to the model.
    pub content: String,
    /// Structured payload for the trace and the client, when the tool has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ToolOutput {
    /// Text-only output.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            data: None,
        }
    }
}

/// Tool failures. Reported to the model as text so it can recover.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Arguments did not match the schema.
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    /// The tool's own failure.
    #[error("{0}")]
    Failed(String),
    /// Ran past its timeout.
    #[error("tool timed out")]
    Timeout,
    /// The request was cancelled.
    #[error("tool cancelled")]
    Cancelled,
}

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
