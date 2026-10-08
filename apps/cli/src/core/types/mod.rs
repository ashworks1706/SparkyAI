//! Data the console passes between its modules: units, statuses, log lines, modes, events.

pub mod compose;
pub mod console;
pub mod error;
pub mod health;
pub mod log;
pub mod sandbox;
pub mod unit;

pub use compose::{ComposePsRow, ServiceState};
pub use console::{Command, Event, Focus, Mode};
pub use error::RunnerError;
pub use health::{Health, Probe};
pub use log::{LogLine, Stream};
pub use sandbox::{
    SANDBOX_SWITCH, SandboxCommand, SandboxReport, SandboxSession, SandboxUnit, session_id,
};
pub use unit::{Group, Kind, Status, Unit};
