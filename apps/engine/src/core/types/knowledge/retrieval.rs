//! RetrievalQuery, RetrievalError.

/// A retrieval request.
#[derive(Debug, Clone)]
pub struct RetrievalQuery {
    /// Natural-language query.
    pub text: String,
    /// How many chunks to return after fusion.
    pub top_k: usize,
    /// Chunks category to search. None searches every category.
    pub category: Option<String>,
}

impl RetrievalQuery {
    /// A query for the top_k closest chunks.
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            top_k,
            category: None,
        }
    }

    /// The same query, narrowed to one chunks category.
    pub fn in_category(self, category: impl Into<String>) -> Self {
        Self {
            category: Some(category.into()),
            ..self
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
