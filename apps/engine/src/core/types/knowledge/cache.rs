//! What the query cache holds for one live query, and why a cache call did not work.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::types::knowledge::query::QueryOutcome;

/// What the cache holds for one query key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Entry {
    /// The answer a query produced, with when it was fetched.
    Answer {
        /// What the source returned.
        outcome: QueryOutcome,
        /// When the fetch that produced it finished.
        fetched_at: DateTime<Utc>,
    },
    /// The refusal a query produced, so a request that waited fails the same way.
    Refusal {
        /// What the scraper said was wrong.
        reason: String,
    },
    /// A request holds the lease and is fetching this query now.
    Pending,
}

/// Why a cache call did not work. A cache failure never fails the request that hit it.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// The cache could not be reached, or answered with something unreadable.
    #[error("query cache: {0}")]
    Backend(String),
}

/// What the cache did for one live query, recorded on the trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheOutcome {
    /// A stored answer within its lifetime was reused.
    Hit,
    /// Another request was already fetching this query, and its answer was shared.
    Coalesced,
    /// Nothing was stored, so this request fetched.
    Miss,
    /// The cache was not reachable, so this request fetched.
    Unavailable,
}
