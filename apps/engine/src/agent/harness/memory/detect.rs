//! The rule detector. A first-person subject next to a stative cue.
//!
//! Generic embeddings encode topic and style rather than whether a sentence is worth
//! remembering, and they place I like this and I do not like this close together, so a cosine
//! gate cannot separate them. A trained head could, which is what the trait is for.

use crate::core::traits::memory::detector::FactDetector;

/// Markers that a sentence is about the person writing it.
pub const SUBJECTS: [&str; 8] = [
    "i ", "i'm ", "i've ", "i'll ", "my ", "me ", "mine ", "myself",
];

/// Cues that a sentence states something rather than asking for something.
pub const CUES: [&str; 26] = [
    "am ",
    "'m ",
    "prefer",
    "like",
    "love",
    "hate",
    "enjoy",
    "want",
    "need",
    "study",
    "studying",
    "major",
    "minor",
    "work",
    "live",
    "taking",
    "enrolled",
    "member",
    "joined",
    "name is",
    "call me",
    "usually",
    "always",
    "never",
    "graduat",
    "interested",
];

/// What the detector accepts.
#[derive(Debug, Clone)]
pub struct Rules {
    /// Lowercase first-person markers.
    pub subjects: Vec<String>,
    /// Lowercase cues that mark a statement.
    pub cues: Vec<String>,
    /// Shortest turn considered.
    pub min_words: usize,
}

impl Default for Rules {
    fn default() -> Self {
        Self::from(&crate::core::config::Detector::default())
    }
}

impl From<&crate::core::config::Detector> for Rules {
    fn from(cfg: &crate::core::config::Detector) -> Self {
        let lower = |v: &Vec<String>, fallback: &[&str]| -> Vec<String> {
            if v.is_empty() {
                fallback.iter().map(|s| (*s).to_owned()).collect()
            } else {
                v.iter().map(|s| s.to_lowercase()).collect()
            }
        };
        Self {
            subjects: lower(&cfg.subjects, &SUBJECTS),
            cues: lower(&cfg.cues, &CUES),
            min_words: cfg.min_words,
        }
    }
}

/// Matches a first-person subject next to a stative cue.
#[derive(Debug, Clone, Default)]
pub struct RuleDetector {
    rules: Rules,
}

impl RuleDetector {
    /// Builds the detector over its rules.
    pub fn new(rules: Rules) -> Self {
        Self { rules }
    }
}

impl FactDetector for RuleDetector {
    fn carries_fact(&self, turn: &str) -> bool {
        let text = turn.trim();
        if text.split_whitespace().count() < self.rules.min_words {
            return false;
        }
        // A question asks for something and states nothing about the asker.
        if text.ends_with('?') {
            return false;
        }
        // Padded so a marker matches a whole word rather than the inside of another.
        let lowered = format!(" {} ", text.to_lowercase().replace(['\n', '\t'], " "));
        let subject = self
            .rules
            .subjects
            .iter()
            .any(|s| lowered.contains(s.as_str()));
        if !subject {
            return false;
        }
        self.rules.cues.iter().any(|c| lowered.contains(c.as_str()))
    }
}
