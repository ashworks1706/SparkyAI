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
    /// Share of agent.history_budget_tokens a compaction keeps as recent turns. The rest is
    /// left for the summary and for the turns that come after it.
    pub keep_share: f64,
}

impl Compaction {
    /// Tokens of recent turns a compaction keeps whole, out of a history budget.
    pub fn keep_tokens(&self, history: usize) -> usize {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let keep = (history as f64 * self.keep_share.clamp(0.0, 1.0)).floor() as usize;
        keep
    }
}

impl Default for Compaction {
    fn default() -> Self {
        Self {
            enabled: true,
            instructions: None,
            max_tokens: 512,
            temperature: 0.0,
            timeout_secs: 30,
            keep_share: 0.4,
        }
    }
}
