//! JSONL trace recording.

use serde::Deserialize;

/// JSONL trace recording. One file per request under dir.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Trace {
    /// Write traces at all.
    pub enabled: bool,
    /// Directory for JSONL traces.
    pub dir: String,
    /// Stop writing the trace of a request past this many bytes. 0 removes the limit.
    pub max_file_bytes: u64,
    /// Delete traces older than this at boot. 0 keeps them forever.
    pub retention_hours: u64,
}

impl Default for Trace {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: ".sparky/traces".into(),
            max_file_bytes: 0,
            retention_hours: 0,
        }
    }
}
