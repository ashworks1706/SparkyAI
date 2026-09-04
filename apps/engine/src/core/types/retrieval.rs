//! `RetrievalQuery`, `RetrievalError`.

/// A retrieval request.
#[derive(Debug, Clone)]
pub struct RetrievalQuery {
    /// Natural-language query.
    pub text: String,
    /// Restrict to these source categories; empty means all.
    pub categories: Vec<String>,
    /// How many chunks to return after fusion.
    pub top_k: usize,
}

impl RetrievalQuery {
    /// A query over every category.
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            categories: Vec::new(),
            top_k,
        }
    }
}

/// Retrieval failures.
#[derive(Debug, thiserror::Error)]
pub enum RetrievalError {
    /// The store could not be queried.
    #[error("retrieval store: {0}")]
    Store(String),
    /// The query could not be embedded.
    #[error("embedding: {0}")]
    Embedding(String),
}
