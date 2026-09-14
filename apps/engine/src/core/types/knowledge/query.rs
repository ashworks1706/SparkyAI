//! QuerySourceInfo, QueryRequest, QueryOutcome: live queries the scraper fetches, engine indexes.

use serde::{Deserialize, Serialize};

/// One parameter a query source accepts, as the scraper published it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryParam {
    /// Parameter name.
    pub name: String,
    /// Whether the query fails without it.
    #[serde(default)]
    pub required: bool,
    /// The only values the scraper accepts, compared without case. Empty accepts any text.
    #[serde(default)]
    pub choices: Vec<String>,
    /// Whether a comma-separated list of choices is accepted.
    #[serde(default)]
    pub many: bool,
}

/// A query source the scraper serves, read from the registry it publishes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuerySourceInfo {
    /// Registry key.
    pub key: String,
    /// Parameters it accepts.
    pub params: Vec<QueryParam>,
    /// Whether the scraper writes a result of this source to the retrieval index.
    #[serde(default = "indexed_by_default")]
    pub indexed: bool,
}

/// A source published before the scraper reported the flag is read as indexed.
fn indexed_by_default() -> bool {
    true
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
    /// The scraper rejected it: unknown source, bad param, unreadable page. Goes back to the model.
    #[error("{0}")]
    Rejected(String),
    /// The scraper did not claim the query in time, so it is not running.
    #[error(
        "the scraper is not running, so {0} cannot be fetched now; start it with just scraper serve"
    )]
    NoWorker(String),
    /// The scraper did not answer inside the budget.
    #[error("the scraper did not answer within {0:?}")]
    Timeout(std::time::Duration),
    /// The request ended before the query did.
    #[error("query cancelled")]
    Cancelled,
}
