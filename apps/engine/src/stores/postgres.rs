//! What the PostgreSQL adapters share: the pool, the error mapping, and the literal helpers.
//!
//! Each adapter lives in a sibling module and is re-exported here.

use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use sqlx::postgres::{PgPool, PgPoolOptions};

use crate::core::types::store::StoreError;

pub use crate::stores::confirmation::PgConfirmations;
pub use crate::stores::conversation::PgConversations;
pub use crate::stores::knowledge::query::PgSourceQueries;
pub use crate::stores::knowledge::retrieval::{PgRetriever, RetrievalTuning};
/// Fusion helpers the core tests exercise directly.
#[cfg(test)]
pub(crate) use crate::stores::knowledge::retrieval::{collapse, rrf};
pub use crate::stores::memory::PgMemory;

/// Opens the pool. Fails fast if the database is unreachable.
pub async fn connect(
    url: &SecretString,
    max_connections: u32,
    acquire_timeout: Duration,
) -> Result<PgPool, StoreError> {
    PgPoolOptions::new()
        .max_connections(max_connections)
        .acquire_timeout(acquire_timeout)
        .connect(url.expose_secret())
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
}

/// Most rows any single query returns.
const MAX_ROWS: i64 = 1_000;

/// Clamps a row limit to MAX_ROWS.
pub(crate) fn row_limit(limit: usize) -> i64 {
    i64::try_from(limit).unwrap_or(MAX_ROWS).min(MAX_ROWS)
}

/// Maps a sqlx error to a StoreError.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn db(e: sqlx::Error) -> StoreError {
    StoreError::Database(e.to_string())
}

/// Quotes a value as a PostgreSQL string literal.
pub(crate) fn quote_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// pgvector text input form: [0.1,0.2,...].
pub(crate) fn vector_literal(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 10 + 2);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&x.to_string());
    }
    s.push(']');
    s
}
