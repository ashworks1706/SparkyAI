//! SandboxRequest, SandboxOutput, SandboxError, and the names the workspace accepts.

use serde::{Deserialize, Serialize};

/// A command to run in an isolated environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxRequest {
    /// The command, run through a shell inside the sandbox.
    pub command: String,
    /// Session to run in. A new name starts one, a running name resumes it, absent runs with none.
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

/// One live session container, as an operator sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxSession {
    /// Container name, which is what kills it.
    pub name: String,
    /// Name the caller gave the session.
    pub session: String,
    /// Seconds since it was started.
    pub age_secs: u64,
    /// Seconds since a command last ran in it.
    pub idle_secs: u64,
    /// Commands run in it.
    pub runs: u64,
}

/// One command the sandbox ran, kept so an operator can watch what the agent is doing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxCommand {
    /// Identifies it across reports, so a reader can follow one command from start to end.
    pub id: u64,
    /// When it started.
    pub at: chrono::DateTime<chrono::Utc>,
    /// Container it ran in, absent for a one-shot container.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    /// Session it ran in, absent for a one-shot container.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    /// The command, as the model wrote it.
    pub command: String,
    /// Exit status. Absent while it runs, and absent after one that timed out or was cancelled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// How long it took. Absent while it is still running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// What the sandbox is doing right now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxReport {
    /// Whether the agent is offered the tool at all.
    pub enabled: bool,
    /// Live session containers, newest use first.
    pub sessions: Vec<SandboxSession>,
    /// Commands the sandbox ran, newest first.
    pub commands: Vec<SandboxCommand>,
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

/// A container-safe session name; refused if empty, too long, or not alnum, hyphen, underscore.
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

/// A workspace file name; refused if empty, too long, or not alnum, hyphen, underscore, dot.
///
/// A name carrying a separator or a parent reference could leave the workspace directory.
pub fn workspace_path(raw: &str) -> Result<String, SandboxError> {
    let name = raw.trim();
    if name.is_empty() || name.len() > 64 {
        return Err(SandboxError::Refused(
            "a workspace name is 1 to 64 characters".into(),
        ));
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
    {
        return Err(SandboxError::Refused(
            "a workspace name takes letters, digits, hyphen, underscore and dot".into(),
        ));
    }
    if name.starts_with('.') || name.contains("..") {
        return Err(SandboxError::Refused(
            "a workspace name does not start with a dot or carry two in a row".into(),
        ));
    }
    Ok(name.to_owned())
}
