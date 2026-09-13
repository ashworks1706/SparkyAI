//! The prompt token estimator.

/// Rough token count for text: its length over chars_per_token, plus a constant for message
/// framing.
pub fn estimate(text: &str, chars_per_token: usize) -> usize {
    text.len() / chars_per_token.max(1) + 4
}
