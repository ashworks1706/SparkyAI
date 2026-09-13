//! FactDetector trait.

/// Decides whether a turn is worth extracting a profile from. Runs on every turn without a model
/// call or network access.
pub trait FactDetector: Send + Sync {
    /// Whether this turn states something about the person who wrote it.
    fn carries_fact(&self, turn: &str) -> bool;
}
