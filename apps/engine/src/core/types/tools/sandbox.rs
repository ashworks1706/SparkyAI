//! SandboxRequest, SandboxOutput, SandboxError.

use serde::{Deserialize, Serialize};

/// A command to run in an isolated environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxRequest {
    /// The command, run through a shell inside the sandbox.
    pub command: String,
    /// Session to run in. A new name starts one; a name already running is resumed, so a file
    /// written under /tmp by an earlier command is still there. Absent runs with no session.
    #[serde(default)]
    pub session: Option<String>,
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
    /// The session it ran in, when it ran in one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
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

/// A session name that can be part of a container name.
///
/// # Errors
/// Returns [`SandboxError::Refused`] when the name is empty, too long, or carries anything but
/// letters, digits, hyphen and underscore.
pub fn session_name(raw: &str) -> Result<String, SandboxError> {
    let name = raw.trim();
    if name.is_empty() || name.len() > 48 {
        return Err(SandboxError::Refused(
            "a session name is 1 to 48 characters".into(),
        ));
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(SandboxError::Refused(
            "a session name takes letters, digits, hyphen and underscore".into(),
        ));
    }
    Ok(name.to_owned())
}
