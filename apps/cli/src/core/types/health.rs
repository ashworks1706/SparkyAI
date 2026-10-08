//! Dependency probes.

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
    /// The engine /health endpoint.
    pub engine: Probe,
    /// The llama-server chat /models endpoint.
    pub model: Probe,
    /// The Phoenix trace UI.
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
