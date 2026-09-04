//! `StoreError`, shared by the conversation and memory stores.

/// Store failures.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The database rejected or could not run the operation.
    #[error("store: {0}")]
    Database(String),
    /// A row the request needs does not exist.
    #[error("not found: {0}")]
    NotFound(String),
}
