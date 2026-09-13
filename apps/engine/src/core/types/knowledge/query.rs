//! QuerySourceInfo, QueryRequest, QueryOutcome: live parameterized source queries.
//!
//! A query source is an ASU site the scraper fetches on demand with parameters the model
//! supplies. What comes back answers one caller and is never retrieval evidence.

use serde::{Deserialize, Serialize};

/// One parameter a query source accepts, as the scraper worker published it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryParam {
    /// Parameter name.
    pub name: String,
    /// Whether the query fails without it.
    #[serde(default)]
    pub required: bool,
    /// The only values the worker accepts, compared without case. Empty accepts any text.
    #[serde(default)]
    pub choices: Vec<String>,
    /// Whether a comma-separated list of choices is accepted.
    #[serde(default)]
    pub many: bool,
}

/// A query source the scraper worker serves, read from the registry it publishes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuerySourceInfo {
    /// Registry key.
    pub key: String,
    /// Parameters it accepts.
    pub params: Vec<QueryParam>,
}

/// One live query to run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Registry key.
    pub source: String,
    /// Parameter values, as strings.
    pub params: serde_json::Map<String, serde_json::Value>,
}

/// What a finished query produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryOutcome {
    /// The source that answered.
    pub source: String,
    /// The URL actually fetched.
    pub url: String,
    /// Readable text of the page.
    pub text: String,
}

/// Why a live query did not answer.
#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    /// The queue could not be reached.
    #[error("query queue: {0}")]
    Store(String),
    /// The worker rejected it: an unknown source, a missing parameter, an unreadable page.
    /// The reason goes back to the model.
    #[error("{0}")]
    Rejected(String),
    /// No worker answered inside the budget.
    #[error("no source worker answered within {0:?}; is `just worker` running?")]
    Timeout(std::time::Duration),
    /// The request ended before the query did.
    #[error("query cancelled")]
    Cancelled,
}
