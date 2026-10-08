//! RunStatus: how a run ended, and what the caller hears about it.

use serde::{Deserialize, Serialize};

/// How a run ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// A final answer was produced.
    Answered,
    /// Stopped to ask the user to confirm an action.
    AwaitingConfirmation,
    /// Hit the step limit.
    StepLimit,
    /// Kept repeating the same tool calls without answering.
    Stalled,
    /// Hit the deadline.
    Deadline,
    /// Cancelled by the caller.
    Cancelled,
    /// The guardrail refused the response.
    Blocked,
    /// Failed with an error.
    Error,
}

impl RunStatus {
    /// What to tell the caller when the loop stopped with no answer. None for Answered and Blocked.
    pub fn explain(&self) -> Option<&'static str> {
        match self {
            Self::Answered | Self::Blocked => None,
            Self::AwaitingConfirmation => Some("I stopped to ask you first."),
            Self::StepLimit => Some("I could not finish within the allowed number of steps."),
            Self::Stalled => {
                Some("I kept repeating myself without getting further; try rephrasing.")
            }
            Self::Deadline => Some("That took too long, so I stopped."),
            Self::Cancelled => Some("Cancelled."),
            Self::Error => Some("Something went wrong before I could answer."),
        }
    }
}
