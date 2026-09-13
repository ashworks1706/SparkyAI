//! Settings that shape the agent loop and the prompt it assembles.

use serde::Deserialize;

use crate::core::config::ConfigError;
use crate::core::types::agent::assemble::{self, Budget};
use crate::core::types::agent::thinking::{ThinkingMode, ThinkingRules, words};
use crate::core::types::trace::progress::{self, ProgressStyle};

/// Agent loop limits. Every field has a default so a bare .env still boots.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Agent {
    /// Model calls per request.
    pub max_steps: u32,
    /// Retries per model call on transport or 5xx errors.
    pub max_model_retries: u32,
    /// Wall-clock budget per request.
    pub request_timeout_secs: u64,
    /// Budget per tool call, unless the tool declares its own.
    pub tool_timeout_secs: u64,
    /// How long a held action waits for caller approval.
    pub confirmation_ttl_secs: u64,
    /// Model calls in flight at once. Match llama-server --parallel. 0 removes the limit.
    pub model_slots: usize,
    /// How long a request waits for a free model slot before reporting the model busy.
    pub model_queue_wait_secs: u64,
    /// Sampling temperature.
    pub temperature: f32,
    /// Prior turns loaded into the prompt.
    pub history_turns: usize,
    /// Memories recalled per request.
    pub memory_recall_limit: usize,
    /// Recall memory and the profile graph for a public request.
    pub recall_in_public: bool,
    /// Whole-prompt token budget.
    pub prompt_budget_tokens: usize,
    /// Cap on the evidence section.
    pub evidence_budget_tokens: usize,
    /// Cap on prior turns.
    pub history_budget_tokens: usize,
    /// Cap on the memory section.
    pub memory_budget_tokens: usize,
    /// Cap on the capabilities section.
    pub capabilities_budget_tokens: usize,
    /// Characters per token the budget estimator assumes.
    pub chars_per_token: usize,
    /// First retry wait, doubled per attempt.
    pub retry_base_ms: u64,
    /// Longest retry wait.
    pub retry_cap_ms: u64,
    /// Longest value recorded on a span; the JSONL trace keeps the rest.
    pub max_span_value_chars: usize,
    /// Characters of a thought, a tool argument list, or a tool result one live progress line
    /// carries.
    pub progress_detail_chars: usize,
    /// When a model call thinks.
    pub thinking: ThinkingRules,
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_model_retries: 2,
            request_timeout_secs: 90,
            tool_timeout_secs: 20,
            confirmation_ttl_secs: 600,
            model_slots: 2,
            model_queue_wait_secs: 30,
            temperature: 0.3,
            history_turns: 20,
            memory_recall_limit: 10,
            recall_in_public: false,
            prompt_budget_tokens: 4_000,
            evidence_budget_tokens: 1_200,
            history_budget_tokens: 1_000,
            memory_budget_tokens: 300,
            capabilities_budget_tokens: 600,
            chars_per_token: 4,
            retry_base_ms: 250,
            retry_cap_ms: 8_000,
            max_span_value_chars: 32_000,
            progress_detail_chars: progress::DETAIL_CHARS,
            thinking: ThinkingRules::default(),
        }
    }
}

impl Agent {
    /// How much detail a live progress line carries.
    pub fn progress_style(&self) -> ProgressStyle {
        ProgressStyle {
            detail_chars: self.progress_detail_chars,
        }
    }

    /// The prompt budgets these settings describe.
    pub fn budget(&self) -> Budget {
        Budget {
            total: self.prompt_budget_tokens,
            evidence: self.evidence_budget_tokens,
            history: self.history_budget_tokens,
            memory: self.memory_budget_tokens,
            capabilities: self.capabilities_budget_tokens,
            chars_per_token: self.chars_per_token,
        }
    }
}

impl Default for ThinkingRules {
    fn default() -> Self {
        Self {
            mode: ThinkingMode::Auto,
            after_tools: true,
            max_quick_chars: 80,
            cues: [
                "why",
                "how",
                "compare",
                "difference",
                "should",
                "plan",
                "explain",
                "which",
            ]
            .map(str::to_owned)
            .to_vec(),
            retry_without: true,
        }
    }
}

/// Rejects a thinking cue that holds no word to match.
///
/// # Errors
/// Returns [ConfigError::Invalid] naming the cue.
pub fn validate_thinking(rules: &ThinkingRules) -> Result<(), ConfigError> {
    match rules.cues.iter().find(|cue| words(cue).is_empty()) {
        Some(cue) => Err(ConfigError::Invalid(format!(
            "agent.thinking.cues holds {cue:?}, which has no word to match"
        ))),
        None => Ok(()),
    }
}

/// The text the harness writes around every prompt. Changing any of it changes the prompt hash.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Prompt {
    /// System instructions. Overrides the built-in default; overridden by system_file.
    pub system: Option<String>,
    /// Path to a file holding the system instructions. Read once at boot.
    pub system_file: Option<String>,
    /// Line naming the user, with {user} and {roles}.
    pub role_line: String,
    /// Line naming a user who holds no roles, with {user}.
    pub role_line_no_roles: String,
    /// Heading above recalled memories.
    pub memory_header: String,
    /// Heading above retrieved evidence.
    pub evidence_header: String,
    /// Line written when retrieval found nothing.
    pub no_evidence_line: String,
    /// Heading above what the model may do.
    pub capabilities_header: String,
    /// Line naming the current date, with {date}.
    pub date_line: String,
    /// Hours from UTC the date is rendered in. Arizona keeps -7 all year.
    pub utc_offset_hours: i32,
}

impl Default for Prompt {
    fn default() -> Self {
        Self {
            system: None,
            system_file: None,
            role_line: assemble::ROLE_LINE.into(),
            role_line_no_roles: assemble::ROLE_LINE_NO_ROLES.into(),
            memory_header: assemble::MEMORY_HEADER.into(),
            evidence_header: assemble::EVIDENCE_HEADER.into(),
            no_evidence_line: assemble::NO_EVIDENCE_LINE.into(),
            capabilities_header: assemble::CAPABILITIES_HEADER.into(),
            date_line: assemble::DATE_LINE.into(),
            utc_offset_hours: -7,
        }
    }
}
