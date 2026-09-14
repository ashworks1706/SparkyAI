//! The rule router: a live cue skips retrieval, a short turn of small talk skips it too.

use crate::core::traits::knowledge::route::Router;
use crate::core::types::knowledge::route::{Route, Skipped};

/// Cues that the answer has to be current, so a live source tool answers it.
pub const LIVE: [&str; 37] = [
    "right now",
    "now",
    "today",
    "tonight",
    "tomorrow",
    "this week",
    "this weekend",
    "currently",
    "current",
    "at the moment",
    "as of now",
    "open now",
    "open today",
    "still open",
    "next shuttle",
    "shuttle",
    "shuttles",
    "study room",
    "study rooms",
    "open seats",
    "seats left",
    "spots left",
    "availability",
    "available now",
    "latest",
    "newest",
    "most recent",
    "breaking",
    "score",
    "scores",
    "final score",
    "who won",
    "happening",
    "upcoming",
    "search the web",
    "on the web",
    "up to date",
];

/// Markers of small talk: a greeting, an acknowledgement, or a question about Sparky itself.
/// Only words that do not also read as part of an ASU question, since a marker skips retrieval.
pub const CHITCHAT: [&str; 31] = [
    "hi",
    "hey",
    "hello",
    "yo",
    "sup",
    "howdy",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "cool",
    "lol",
    "lmao",
    "haha",
    "bye",
    "goodbye",
    "see ya",
    "how are you",
    "whats up",
    "what s up",
    "who are you",
    "what are you",
    "what can you do",
    "nevermind",
    "never mind",
    "no worries",
];

/// What the router accepts.
#[derive(Debug, Clone)]
pub struct Rules {
    /// Normalized cues that the answer has to be current.
    pub live: Vec<String>,
    /// Normalized markers of small talk.
    pub chitchat: Vec<String>,
    /// Longest turn, in words, a chitchat marker may skip retrieval for.
    pub max_chitchat_words: usize,
}

impl Default for Rules {
    fn default() -> Self {
        Self::from(&crate::core::config::Router::default())
    }
}

impl From<&crate::core::config::Router> for Rules {
    fn from(cfg: &crate::core::config::Router) -> Self {
        let phrases = |set: &[String], fallback: &[&str]| -> Vec<String> {
            let given: Vec<String> = set.iter().map(|s| normalize(s)).collect();
            if given.iter().all(String::is_empty) {
                fallback.iter().map(|s| normalize(s)).collect()
            } else {
                given.into_iter().filter(|s| !s.is_empty()).collect()
            }
        };
        Self {
            live: phrases(&cfg.live, &LIVE),
            chitchat: phrases(&cfg.chitchat, &CHITCHAT),
            max_chitchat_words: cfg.max_chitchat_words,
        }
    }
}

/// Lowercase words separated by single spaces; every other character becomes a separator.
fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = true;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            out.extend(ch.to_lowercase());
            space = false;
        } else if !space {
            out.push(' ');
            space = true;
        }
    }
    out.trim_end().to_owned()
}

/// Whether phrase appears in normalized text on word boundaries.
fn holds(padded: &str, phrase: &str) -> bool {
    padded.contains(&format!(" {phrase} "))
}

/// Skips retrieval for a live cue, and for a short turn that is only small talk.
#[derive(Debug, Clone, Default)]
pub struct RuleRouter {
    rules: Rules,
}

impl RuleRouter {
    /// Builds the router over its rules.
    pub fn new(rules: Rules) -> Self {
        Self { rules }
    }
}

impl Router for RuleRouter {
    fn route(&self, input: &str) -> Route {
        let text = normalize(input);
        if text.is_empty() {
            return Route::Skip(Skipped::Chitchat);
        }
        let padded = format!(" {text} ");
        if self.rules.live.iter().any(|cue| holds(&padded, cue)) {
            return Route::Skip(Skipped::Live);
        }
        let words = text.split(' ').count();
        if words <= self.rules.max_chitchat_words
            && self
                .rules
                .chitchat
                .iter()
                .any(|marker| holds(&padded, marker))
        {
            return Route::Skip(Skipped::Chitchat);
        }
        Route::Retrieve
    }
}
