//! FactDetector trait.

/// Decides whether a turn is worth extracting a profile from.
///
/// The gate runs on every turn, so it takes no model call and no network. Extraction is what
/// costs; this only decides whether extraction is worth starting.
pub trait FactDetector: Send + Sync {
    /// Whether this turn states something about the person who wrote it.
    fn carries_fact(&self, turn: &str) -> bool;
}
