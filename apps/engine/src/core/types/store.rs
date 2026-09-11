//! StoreError, shared by the conversation and memory stores.

/// Store failures.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The database rejected or could not run the operation.
    #[error("store: {0}")]
    Database(String),
    /// The conversation exists and belongs to another user or tenant.
    #[error("store: conversation belongs to another caller")]
    NotOwned,
}
