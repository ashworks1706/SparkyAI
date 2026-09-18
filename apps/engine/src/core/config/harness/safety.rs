//! What the policy allows, what the guardrail refuses, and how a sandboxed command runs.

use serde::Deserialize;

use crate::core::types::tools::RiskClass;

/// What the risk policy allows, denies, and holds for confirmation.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Policy {
    /// Roles allowed to run external_write and above. Empty denies everyone.
    pub write_roles: Vec<String>,
    /// Let tools read inside the authenticated session of the user.
    pub allow_authenticated_reads: bool,
    /// Lowest risk class that must be confirmed before it runs.
    pub confirm_from: RiskClass,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            write_roles: vec!["MANAGE_GUILD".into()],
            allow_authenticated_reads: false,
            confirm_from: RiskClass::ExternalWrite,
        }
    }
}

/// The gate every model response passes.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Guardrail {
    /// Check responses at all.
    pub enabled: bool,
    /// Phrases that block a response wherever they appear. Matched without case.
    pub denied_phrases: Vec<String>,
    /// Longest answer allowed. 0 removes the limit.
    pub max_answer_chars: usize,
    /// Shown in place of a blocked response.
    pub replacement: String,
}

impl Default for Guardrail {
    fn default() -> Self {
        Self {
            enabled: true,
            denied_phrases: Vec::new(),
            max_answer_chars: 8_000,
            replacement: "I cannot answer that. Ask a moderator if you need help.".into(),
        }
    }
}

/// A command in an isolated environment.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct SandboxSettings {
    /// Offer the sandbox to the model. The engine needs access to the container runtime.
    pub enabled: bool,
    /// Refuse to start when the runtime is unreachable. Off leaves the tool unregistered instead.
    pub required: bool,
    /// Container runtime binary.
    pub runtime: String,
    /// Image the command runs in.
    pub image: String,
    /// Memory ceiling.
    pub memory: String,
    /// CPU ceiling.
    pub cpus: String,
    /// Process ceiling.
    pub pids: u32,
    /// Wall-clock budget for one command.
    pub timeout_secs: u64,
    /// Longest stdout or stderr handed back to the model.
    pub max_output_chars: usize,
    /// Risk class the tool declares to Policy.
    pub risk: RiskClass,
    /// How long a session container stays up with nothing running in it.
    pub session_idle_secs: u64,
    /// Sessions one user may hold at once. Starting one past the cap reaps the least recently used.
    pub max_sessions: usize,
    /// Size of the writable workspace, in mebibytes.
    pub workspace_mb: u32,
    /// What the tool tells the model it is for.
    pub description: String,
    /// What the tool tells the model the command argument is.
    pub command_description: String,
    /// What the tool tells the model the session argument is.
    pub session_description: String,
}

impl Default for SandboxSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            required: false,
            runtime: "docker".into(),
            image: "ghcr.io/ashworks1706/sparkyai-sandbox:main".into(),
            memory: "256m".into(),
            cpus: "1".into(),
            pids: 128,
            timeout_secs: 20,
            max_output_chars: 4_000,
            risk: RiskClass::PrepareWrite,
            session_idle_secs: 900,
            max_sessions: 4,
            workspace_mb: 64,
            description: DESCRIPTION.into(),
            command_description: COMMAND_DESCRIPTION.into(),
            session_description: SESSION_DESCRIPTION.into(),
        }
    }
}

/// What the sandbox tells the model it is for, when sandbox.description is unset.
pub const DESCRIPTION: &str = "Run a shell command in an isolated environment. Use it to work \
    something out rather than doing it in your head: arithmetic, dates, sorting, filtering, or \
    reading a file an earlier tool left in the workspace. python3, jq and the usual text tools \
    are installed. There is no network, so it cannot fetch anything or reach ASU.";

/// What the sandbox tells the model the command argument is.
pub const COMMAND_DESCRIPTION: &str = "The shell command to run.";

/// What the sandbox tells the model the session argument is.
pub const SESSION_DESCRIPTION: &str = "Name a session to keep files in the workspace between \
    calls. Reuse the same name to resume it. Results an earlier tool wrote to the workspace name \
    the session they are in.";
