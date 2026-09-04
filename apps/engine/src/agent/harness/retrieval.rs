//! `Retriever`, `Embedder`, `Reranker` traits and their query types.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;

/// A retrieval request.
#[derive(Debug, Clone)]
pub struct RetrievalQuery {
    /// Natural-language query.
    pub text: String,
    /// Restrict to these source categories; empty means all.
    pub categories: Vec<String>,
    /// How many chunks to return after reranking.
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
    /// Reranking failed; callers may fall back to fused order.
    #[error("rerank: {0}")]
    Rerank(String),
}

/// Finds evidence for a query, scoped to the request's tenant.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Returns evidence best first.
    async fn retrieve(
        &self,
        ctx: &RequestContext,
        query: &RetrievalQuery,
    ) -> Result<Vec<Evidence>, RetrievalError>;
}

/// Turns text into vectors, with the same model the index was built with.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embeds one or more texts, one vector each, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, RetrievalError>;
    /// Vector dimension.
    fn dim(&self) -> usize;
}

/// Scores documents against a query.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Returns one score per document, same order. Ordering only; not calibrated.
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, RetrievalError>;
}
