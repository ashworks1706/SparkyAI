//! The prompt token estimator.
//!
//! One function, used by everything that has to fit inside the prompt budget.

/// Rough token count for text.
///
/// chars_per_token is the divisor, set per tokenizer. The constant addend covers the
/// per-message framing every provider adds.
pub fn estimate(text: &str, chars_per_token: usize) -> usize {
    text.len() / chars_per_token.max(1) + 4
}
