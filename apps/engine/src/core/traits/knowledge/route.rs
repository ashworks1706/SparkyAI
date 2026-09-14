//! Router: the gate retrieval passes before it runs.

use crate::core::types::knowledge::route::Route;

/// Decides whether a question needs retrieval. Runs on every turn, so it never calls a model.
pub trait Router: Send + Sync {
    /// The route for one user input.
    fn route(&self, input: &str) -> Route;
}
