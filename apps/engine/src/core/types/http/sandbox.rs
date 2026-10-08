//! Wire shapes of the sandbox routes.

use serde::{Deserialize, Serialize};

/// What POST /sandbox/enabled takes.
#[derive(Debug, Serialize, Deserialize)]
pub struct Switch {
    /// Whether the agent is offered the tool.
    pub enabled: bool,
}

/// What a kill answers with.
#[derive(Debug, Serialize)]
pub struct Killed {
    /// The container that was removed.
    pub name: String,
}
