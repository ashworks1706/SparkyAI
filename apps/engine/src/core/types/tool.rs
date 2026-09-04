//! `ToolDefinition`, `RiskClass`, `ToolOutput`, `ToolError`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    /// Holds state between calls (a browser, a session). Calls to it never run in parallel
    /// with other calls in the same step.
    #[serde(default)]
    pub sequential: bool,
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
