//! Data the console passes between its modules: units, statuses, log lines, modes, events.

use std::collections::HashMap;

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Sidebar section a unit belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    /// Datastores and Phoenix.
    Infra,
    /// llama-server chat and embed.
    Models,
    /// Firecrawl and Playwright MCP.
    Tools,
    /// Long-running host processes.
    Apps,
    /// One-shot recipes.
    Tasks,
    /// Compose stacks and images, for a dev box or the `RunPod` host.
    Deploy,
}

impl Group {
    /// Sidebar heading.
    pub fn title(self) -> &'static str {
        match self {
            Self::Infra => "infra",
            Self::Models => "models",
            Self::Tools => "tools",
            Self::Apps => "apps",
            Self::Tasks => "tasks",
            Self::Deploy => "deploy",
        }
    }

    /// Display order.
    pub const ALL: [Self; 6] = [
        Self::Infra,
        Self::Models,
        Self::Tools,
        Self::Apps,
        Self::Tasks,
        Self::Deploy,
    ];
}

/// How a unit is run and stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A docker compose service; `profile` gates optional ones.
    Service {
        /// Compose service name.
        service: String,
        /// Compose profile, if the service needs one.
        profile: Option<String>,
    },
    /// A long-running host process started through a just recipe.
    Process,
    /// A just recipe that runs to completion.
    Task,
}

/// Something the console can start, stop, and watch.
#[derive(Debug, Clone)]
pub struct Unit {
    /// Stable name shown in the sidebar and used in commands.
    pub id: String,
    /// Sidebar section.
    pub group: Group,
    /// How to run it.
    pub kind: Kind,
    /// Arguments after `just` for processes and tasks.
    pub args: Vec<String>,
    /// One-line description.
    pub hint: String,
    /// Where it listens, if it does.
    pub url: Option<String>,
}

impl Unit {
    /// Compose service name, for services.
    pub fn service(&self) -> Option<&str> {
        match &self.kind {
            Kind::Service { service, .. } => Some(service),
            _ => None,
        }
    }
}

/// Where a unit is in its lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Not running.
    Stopped,
    /// Start requested; no confirmation yet.
    Starting,
    /// Up.
    Running,
    /// Ran and exited with this code.
    Exited(i32),
    /// Could not be started or watched.
    Failed(String),
}

impl Status {
    /// Whether stop makes sense.
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Starting | Self::Running)
    }

    /// Single-character marker for the sidebar.
    pub fn glyph(&self) -> &'static str {
        match self {
            Self::Stopped => "○",
            Self::Starting => "◐",
            Self::Running => "●",
            Self::Exited(0) => "✓",
            Self::Exited(_) => "✗",
            Self::Failed(_) => "!",
        }
    }

    /// Short label for the log pane title.
    pub fn label(&self) -> String {
        match self {
            Self::Stopped => "stopped".into(),
            Self::Starting => "starting".into(),
            Self::Running => "running".into(),
            Self::Exited(code) => format!("exit {code}"),
            Self::Failed(why) => format!("failed: {why}"),
        }
    }
}

/// Which output a log line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    /// Child stdout.
    Out,
    /// Child stderr.
    Err,
    /// The console's own note about the unit.
    Meta,
}

/// One captured line.
#[derive(Debug, Clone)]
pub struct LogLine {
    /// When it was captured.
    pub at: DateTime<Local>,
    /// Source.
    pub stream: Stream,
    /// Content without the trailing newline.
    pub text: String,
}

impl LogLine {
    /// A line captured now.
    pub fn now(stream: Stream, text: impl Into<String>) -> Self {
        Self {
            at: Local::now(),
            stream,
            text: text.into(),
        }
    }
}

/// Input mode, vim-style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Keys navigate and act.
    Normal,
    /// Typing a `:` command.
    Command,
    /// Typing a `/` search over the selected unit's logs.
    Search,
    /// Typing a chat message to the agent.
    Chat,
}

/// Which pane keys act on in normal mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    /// The unit list.
    Units,
    /// The log pane.
    Logs,
    /// The chat transcript.
    Chat,
}

/// One row of `docker compose ps`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceState {
    /// `running`, `exited`, `created`, `restarting`, `paused`, `dead`.
    pub state: String,
    /// `healthy`, `unhealthy`, `starting`, or empty without a healthcheck.
    pub health: String,
    /// Last exit code.
    pub exit_code: i32,
}

impl ServiceState {
    /// Maps a compose row onto the console's status.
    pub fn status(&self) -> Status {
        match (self.state.as_str(), self.health.as_str()) {
            ("running", "unhealthy") => Status::Failed("unhealthy".into()),
            ("running", "starting") | ("created" | "restarting", _) => Status::Starting,
            ("running", _) => Status::Running,
            ("exited", _) if self.exit_code == 0 => Status::Stopped,
            ("exited" | "dead", _) => Status::Exited(self.exit_code),
            (other, _) => Status::Failed(other.to_owned()),
        }
    }
}

/// Result of one dependency probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Probe {
    /// Not checked yet.
    Unknown,
    /// Reachable and healthy.
    Up,
    /// Reachable but reporting a problem, with its message.
    Degraded(String),
    /// Not reachable.
    Down,
}

/// Liveness of the things the agent depends on.
#[derive(Debug, Clone)]
pub struct Health {
    /// Engine `/health`.
    pub engine: Probe,
    /// llama-server chat `/models`.
    pub model: Probe,
    /// Phoenix UI.
    pub phoenix: Probe,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            engine: Probe::Unknown,
            model: Probe::Unknown,
            phoenix: Probe::Unknown,
        }
    }
}

/// What the console sends to `/chat`. Mirrors the engine's `ChatRequest`.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    /// Caller id.
    pub user_id: String,
    /// Channel label.
    pub channel_id: String,
    /// Roles to assert.
    pub roles: Vec<String>,
    /// Continue this conversation, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<Uuid>,
    /// The question.
    pub message: String,
}

/// A pending confirmation the engine wants approved.
#[derive(Debug, Clone, Deserialize)]
pub struct Confirmation {
    /// Tool that would run.
    pub tool: String,
    /// What would happen.
    pub summary: String,
}

/// What `/chat` returns. Mirrors the engine's `ChatResponse`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    /// Trace id.
    pub request_id: Uuid,
    /// Conversation to continue with.
    pub conversation_id: Uuid,
    /// The answer.
    pub text: String,
    /// Citation lines.
    #[serde(default)]
    pub citations: Vec<String>,
    /// Set when the engine stopped to ask.
    #[serde(default)]
    pub confirmation: Option<Confirmation>,
    /// How the run ended.
    pub status: String,
    /// Loop steps taken.
    #[serde(default)]
    pub steps: u32,
    /// Tokens used.
    #[serde(default)]
    pub tokens: u64,
}

/// One entry in the chat transcript.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    /// Who said it.
    pub role: Role,
    /// Body.
    pub text: String,
    /// Citations, for agent turns.
    pub citations: Vec<String>,
    /// Status line under an agent turn: request id, status, steps, tokens, latency.
    pub meta: Option<String>,
}

/// Speaker in the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The developer at the keyboard.
    User,
    /// The engine.
    Agent,
    /// The console itself (errors, notes).
    System,
}

/// Everything that can wake the UI loop.
#[derive(Debug)]
pub enum Event {
    /// A key press.
    Key(crossterm::event::KeyEvent),
    /// Redraw timer.
    Tick,
    /// Terminal resized.
    Resize,
    /// A unit produced a line.
    Log {
        /// Unit id.
        unit: String,
        /// The line.
        line: LogLine,
    },
    /// A process or task ended.
    Exited {
        /// Unit id.
        unit: String,
        /// Exit code, if the process was not killed by a signal.
        code: Option<i32>,
    },
    /// Fresh compose service states, keyed by service name.
    Services(HashMap<String, ServiceState>),
    /// Fresh dependency probes.
    Health(Health),
    /// The engine answered (or failed to).
    ChatReply {
        /// Round-trip time.
        latency_ms: u64,
        /// The reply or the failure text.
        result: Result<ChatResponse, String>,
    },
}

/// Parsed `:` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Leave the console, stopping host processes.
    Quit,
    /// Start a unit by id.
    Start(String),
    /// Stop a unit by id.
    Stop(String),
    /// Stop then start a unit by id.
    Restart(String),
    /// Send a chat message.
    Ask(String),
    /// Set the roles asserted on chat requests.
    Roles(Vec<String>),
    /// Start a new conversation.
    Reset,
    /// Run an arbitrary just recipe as an ad-hoc task.
    Just(Vec<String>),
    /// Show the key map.
    Help,
    /// Clear the selected unit's logs.
    Clear,
    /// Not understood.
    Unknown(String),
}
