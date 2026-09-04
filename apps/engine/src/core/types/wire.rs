//! Tool argument schemas.

use serde::Deserialize;

/// Arguments the model passes to `search_asu`.
#[derive(Deserialize)]
pub struct SearchArgs {
    /// What to look for.
    pub query: String,
    /// Optional source categories to restrict to.
    #[serde(default)]
    pub categories: Vec<String>,
}
