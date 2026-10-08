//! SandboxRequest, SandboxOutput, SandboxError, and the names the workspace accepts.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::core::config::SandboxSettings;

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

/// Port the egress proxy listens on.
pub const PROXY_PORT: u16 = 3128;

/// The way out of the sandbox when commands may reach the public internet.
#[derive(Debug, Clone)]
pub struct Egress {
    /// Internal network the containers join. It has no route out of its own.
    pub network: String,
    /// Image of the proxy that joins the network and the outside.
    pub proxy_image: String,
    /// Container name of the proxy, its host name on the network.
    pub proxy_name: String,
}

impl Egress {
    /// The proxy address the containers are handed.
    pub(crate) fn proxy_url(&self) -> String {
        format!("http://{}:{PROXY_PORT}", self.proxy_name)
    }
}

/// How the sandbox is started and what it may consume.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Container runtime binary.
    pub runtime: String,
    /// Image the command runs in.
    pub image: String,
    /// Memory ceiling, in the form the runtime accepts.
    pub memory: String,
    /// CPU ceiling, in the form the runtime accepts.
    pub cpus: String,
    /// Process ceiling.
    pub pids: u32,
    /// Wall-clock budget.
    pub timeout: Duration,
    /// Longest stdout or stderr handed back.
    pub max_output_chars: usize,
    /// How long a session container stays up with nothing running in it.
    pub session_idle_secs: u64,
    /// Sessions one caller may hold at once.
    pub max_sessions: usize,
    /// Session containers across every caller.
    pub max_sessions_total: usize,
    /// Commands running at once across every caller.
    pub max_running: usize,
    /// Label value naming this engine's containers apart from another engine's.
    pub instance: String,
    /// Size of the writable workspace, in mebibytes.
    pub workspace_mb: u32,
    /// Commands kept for the operator view.
    pub recent_commands: usize,
    /// The way out to the public internet. None runs with no network.
    pub egress: Option<Egress>,
}

impl Default for Limits {
    fn default() -> Self {
        Self::from(&SandboxSettings::default())
    }
}

impl From<&SandboxSettings> for Limits {
    fn from(cfg: &SandboxSettings) -> Self {
        Self {
            runtime: cfg.runtime.clone(),
            image: cfg.image.clone(),
            memory: cfg.memory.clone(),
            cpus: cfg.cpus.clone(),
            pids: cfg.pids,
            timeout: Duration::from_secs(cfg.timeout_secs),
            max_output_chars: cfg.max_output_chars,
            session_idle_secs: cfg.session_idle_secs,
            max_sessions: cfg.max_sessions,
            max_sessions_total: cfg.max_sessions_total,
            max_running: cfg.max_running,
            instance: cfg.instance.clone(),
            workspace_mb: cfg.workspace_mb,
            recent_commands: cfg.recent_commands,
            egress: cfg.egress.then(|| Egress {
                network: cfg.egress_network.clone(),
                proxy_image: cfg.egress_proxy_image.clone(),
                proxy_name: cfg.egress_proxy_name.clone(),
            }),
        }
    }
}
