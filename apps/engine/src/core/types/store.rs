//! `StoreError`, shared by the conversation and memory stores.

/// Store failures.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The database rejected or could not run the operation.
    #[error("store: {0}")]
    Database(String),
}
