//! FactDetector trait.

/// Decides if a turn is worth extracting a profile from. Runs every turn, no model call or network.
pub trait FactDetector: Send + Sync {
    /// Whether this turn states something about the person who wrote it.
    fn carries_fact(&self, turn: &str) -> bool;
}
