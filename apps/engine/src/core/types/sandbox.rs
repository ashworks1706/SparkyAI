//! SandboxRequest, SandboxOutput, SandboxError.

use serde::{Deserialize, Serialize};

/// A command to run in an isolated environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxRequest {
    /// The command, run through a shell inside the sandbox.
    pub command: String,
}

/// What a sandboxed command produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxOutput {
    /// Exit status. Non-zero is reported, not treated as a failure of the tool.
    pub exit_code: i32,
    /// Standard output, truncated to the configured limit.
    pub stdout: String,
    /// Standard error, truncated to the configured limit.
    pub stderr: String,
}

/// Why a sandboxed command did not run.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// The runtime could not be started.
    #[error("sandbox runtime: {0}")]
    Runtime(String),
    /// The command ran past its budget.
    #[error("sandbox timed out")]
    Timeout,
    /// The request was refused before anything ran.
    #[error("{0}")]
    Refused(String),
}
