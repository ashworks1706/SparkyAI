//! Docker compose service states.

use super::unit::Status;

/// One row of docker compose ps --format json, as printed.
#[derive(Debug, serde::Deserialize)]
pub struct ComposePsRow {
    /// Service name.
    #[serde(rename = "Service")]
    pub service: String,
    /// Container state.
    #[serde(rename = "State")]
    pub state: String,
    /// Healthcheck result, empty without a healthcheck.
    #[serde(rename = "Health", default)]
    pub health: String,
    /// Last exit code.
    #[serde(rename = "ExitCode", default)]
    pub exit_code: i32,
}

/// One row of docker compose ps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceState {
    /// One of running, exited, created, restarting, paused, or dead.
    pub state: String,
    /// One of healthy, unhealthy, starting, or empty without a healthcheck.
    pub health: String,
    /// Last exit code.
    pub exit_code: i32,
}

impl ServiceState {
    /// Maps a compose row onto the console status.
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
