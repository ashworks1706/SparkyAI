//! How history that no longer fits is compacted.

use serde::Deserialize;

/// The chat agent. Replaces the turns that no longer fit with one turn that does.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Compaction {
    /// Compact at all. Off leaves history trimmed by dropping the oldest turns.
    pub enabled: bool,
    /// Instructions for the chat agent. Empty uses the built-in default.
    pub instructions: Option<String>,
    /// Completion budget for the compacted turn.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Wall-clock budget for the call.
    pub timeout_secs: u64,
}

impl Default for Compaction {
    fn default() -> Self {
        Self {
            enabled: true,
            instructions: None,
            max_tokens: 512,
            temperature: 0.0,
            timeout_secs: 30,
        }
    }
}
