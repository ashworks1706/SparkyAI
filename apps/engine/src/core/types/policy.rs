//! `ProposedAction`, `ConfirmationRequest`, `Decision`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::core::types::tool::RiskClass;

/// A tool call the model wants to make, before it runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedAction {
    /// Tool name.
    pub tool: String,
    /// Declared risk of that tool.
    pub risk: RiskClass,
    /// Exact arguments. A confirmation binds to these bytes.
    pub arguments: Value,
}

/// What the user must approve.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmationRequest {
    /// Single-use token the client echoes back.
    pub token: Uuid,
    /// Tool name.
    pub tool: String,
    /// Hash of the exact arguments; a changed payload needs a new confirmation.
    pub payload_hash: String,
    /// Plain-language statement of what happens, where, with what data, and whether it reverses.
    pub summary: String,
}

/// A confirmation the engine is holding, with everything needed to run it once approved.
///
/// The caller never sends the arguments back: they are read from here, so an approval cannot
/// change what it approves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingAction {
    /// Provider call id the tool result must answer, so the loop can carry on from it.
    pub call_id: String,
    /// What runs on approval.
    pub action: ProposedAction,
}

/// The policy verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    /// Run it.
    Allow,
    /// Do not run it; the model is told why.
    Deny {
        /// Reason shown to the model and recorded in the trace.
        reason: String,
    },
    /// Stop and ask the user first.
    Confirm(ConfirmationRequest),
}
