//! The engine sandbox: its rows in the sidebar and the report the engine serves.

use chrono::DateTime;

/// Which part of the engine sandbox a row stands for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxUnit {
    /// Whether the agent is offered the sandbox at all.
    Switch,
    /// One live session container.
    Session {
        /// Container name, which is what kills it.
        container: String,
    },
}

/// Unit id of the sandbox switch.
pub const SANDBOX_SWITCH: &str = "sandbox";

/// The unit id one session container is listed under.
pub fn session_id(container: &str) -> String {
    format!("sandbox:{container}")
}

/// One live session container, as the engine reports it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct SandboxSession {
    /// Container name.
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

/// One command the sandbox ran or is running.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct SandboxCommand {
    /// Identifies it across reports.
    pub id: u64,
    /// When it started.
    pub at: DateTime<chrono::Utc>,
    /// Container it ran in, absent for a one-shot container.
    #[serde(default)]
    pub container: Option<String>,
    /// Session it ran in, absent for a one-shot container.
    #[serde(default)]
    pub session: Option<String>,
    /// The command, as the model wrote it.
    pub command: String,
    /// Exit status, absent while it is still running.
    #[serde(default)]
    pub exit_code: Option<i32>,
    /// How long it took, absent while it is still running.
    #[serde(default)]
    pub duration_ms: Option<u64>,
}

impl SandboxCommand {
    /// Whether it has ended, however it ended.
    pub fn ended(&self) -> bool {
        self.duration_ms.is_some()
    }

    /// The log line the console shows for it.
    pub fn line(&self) -> String {
        let where_ = match &self.session {
            Some(session) => format!("[{session}]"),
            None => "[one-shot]".to_owned(),
        };
        match (self.exit_code, self.duration_ms) {
            (Some(code), Some(ms)) => format!("{where_} {} -> exit {code} in {ms}ms", self.command),
            (None, Some(ms)) => {
                format!(
                    "{where_} {} -> ended with no status after {ms}ms",
                    self.command
                )
            }
            _ => format!("{where_} {} -> running", self.command),
        }
    }
}

/// What the engine sandbox is doing right now.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
pub struct SandboxReport {
    /// Whether the agent is offered the tool at all.
    pub enabled: bool,
    /// Live session containers, least idle first.
    pub sessions: Vec<SandboxSession>,
    /// Commands the sandbox ran, newest first.
    pub commands: Vec<SandboxCommand>,
}
