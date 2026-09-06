//! The prompt token estimator.
//!
//! One function, used by everything that has to fit inside the prompt budget. Three copies of
//! this arithmetic drifted apart once; a budget computed one way and spent another silently
//! overfills the context window.

/// Rough token count for `text`.
///
/// `chars_per_token` is a setting because the right divisor depends on the tokenizer: URLs and
/// ASU course codes split far more finely than prose. The constant addend covers the per-message
/// framing every provider adds.
pub fn estimate(text: &str, chars_per_token: usize) -> usize {
    text.len() / chars_per_token.max(1) + 4
}
