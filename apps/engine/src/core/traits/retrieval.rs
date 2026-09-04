//! `Retriever` and `Embedder` traits.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::retrieval::{RetrievalError, RetrievalQuery};

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
