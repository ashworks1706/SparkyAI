//! Verdict and GuardrailError. What the gate every response passes may answer.

use serde::{Deserialize, Serialize};

/// What the guardrail decided about a response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// The response proceeds unchanged.
    Pass,
    /// The response is replaced by this text and the run ends.
    Block {
        /// Shown to the user in place of the response.
        replacement: String,
        /// Why it was blocked, for the trace.
        reason: String,
    },
}

/// Which branch of the loop a response is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// The model asked to run capabilities.
    Capability,
    /// The model answered the user.
    Answer,
}

impl Stage {
    /// Name this stage records under.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Capability => "capability",
            Self::Answer => "answer",
        }
    }
}
