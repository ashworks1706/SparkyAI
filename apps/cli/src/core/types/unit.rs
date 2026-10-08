//! Units the console runs: their group, how they run, and where they are in their lifecycle.

use super::sandbox::SandboxUnit;

/// Sidebar section a unit belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    /// Datastores, the trace UI, and metrics.
    Infra,
    /// llama-server chat and embed.
    Models,
    /// Firecrawl.
    Tools,
    /// Long-running host processes.
    Apps,
    /// Containers the engine runs commands in.
    Sandboxes,
    /// One-shot recipes.
    Tasks,
    /// Compose stacks and images, for a dev box or the RunPod host.
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
            Self::Sandboxes => "sandboxes",
            Self::Tasks => "tasks",
            Self::Deploy => "deploy",
        }
    }

    /// Display order.
    pub const ALL: [Self; 7] = [
        Self::Infra,
        Self::Models,
        Self::Tools,
        Self::Apps,
        Self::Sandboxes,
        Self::Tasks,
        Self::Deploy,
    ];
}

/// How a unit is run and stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A docker compose service. The profile field gates optional ones.
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
    /// Something the engine owns, driven over its HTTP surface.
    Sandbox(SandboxUnit),
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
    /// Arguments after just for processes and tasks.
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
    /// Start requested, with no confirmation yet.
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

    /// Running when on, stopped when off.
    pub fn from_on(on: bool) -> Self {
        if on { Self::Running } else { Self::Stopped }
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
