//! Whether a model call reasons before it answers, and why.

use serde::{Deserialize, Serialize};

/// How the loop sets thinking on each model call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingMode {
    /// Every call thinks.
    On,
    /// No call thinks.
    Off,
    /// The rules decide per call.
    Auto,
}

/// Rules deciding thinking per model call, read from agent.thinking. Default lives in core::config.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ThinkingRules {
    /// How thinking is set.
    pub mode: ThinkingMode,
    /// Think on a step that has tool results to read.
    pub after_tools: bool,
    /// Longest question, in characters, answered from evidence without thinking.
    pub max_quick_chars: usize,
    /// Words or phrases that make a question think, matched whole and case-insensitively.
    pub cues: Vec<String>,
    /// Ask again without thinking when thinking left no answer.
    pub retry_without: bool,
}

/// Why a model call did or did not think.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingReason {
    /// The mode is on or off.
    Mode,
    /// The call is offered no tools and only answers.
    AnswerOnly,
    /// The step has tool results to read.
    AfterTools,
    /// The question holds a cue.
    Cue,
    /// The question is longer than a quick one.
    Long,
    /// Retrieval found nothing, so the model plans a tool call.
    NoEvidence,
    /// A short question with evidence in the prompt.
    Quick,
    /// Thinking left no answer, so the call is made again without it.
    Retry,
}

impl ThinkingReason {
    /// The snake case name recorded on a span.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mode => "mode",
            Self::AnswerOnly => "answer_only",
            Self::AfterTools => "after_tools",
            Self::Cue => "cue",
            Self::Long => "long",
            Self::NoEvidence => "no_evidence",
            Self::Quick => "quick",
            Self::Retry => "retry",
        }
    }
}

/// Whether one model call thinks, and the rule that said so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThinkingChoice {
    /// The call thinks.
    pub on: bool,
    /// The rule that decided.
    pub reason: ThinkingReason,
}

/// The lowercase words of text, split on anything that is not a letter or a digit.
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}
