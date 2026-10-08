//! Console errors.

/// Runner failures.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    /// The child could not be spawned.
    #[error("spawn `{cmd}`: {source}")]
    Spawn {
        /// Command line attempted.
        cmd: String,
        /// OS error.
        source: std::io::Error,
    },
}
