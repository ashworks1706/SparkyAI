//! Console state and input: modes, focus, the events that wake the loop, and commands.

use std::collections::HashMap;

use super::compose::ServiceState;
use super::health::Health;
use super::log::LogLine;
use super::sandbox::SandboxReport;

/// Input mode, vim-style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Keys navigate and act.
    Normal,
    /// Typing a command after a colon.
    Command,
    /// Typing a slash search over the logs of the selected unit.
    Search,
}

/// Which pane keys act on in normal mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    /// The unit list.
    Units,
    /// The log pane.
    Logs,
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
    /// Fresh compose service states keyed by service name, or the docker compose ps error.
    Services(Result<HashMap<String, ServiceState>, String>),
    /// Fresh dependency probes.
    Health(Health),
    /// Fresh sandbox report, or why it could not be read.
    Sandbox(Result<SandboxReport, String>),
    /// An action taken on the engine sandbox finished.
    SandboxActed(Result<String, String>),
    /// The terminal stopped delivering input.
    InputLost(String),
}

/// A parsed command line.
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
    /// Run an arbitrary just recipe as an ad-hoc task.
    Just(Vec<String>),
    /// Show the key map.
    Help,
    /// Clear the logs of the selected unit.
    Clear,
    /// Not understood.
    Unknown(String),
}
