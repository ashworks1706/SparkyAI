//! RigEmbedder: the Embedder over the OpenAI-compatible embeddings model of Rig.

use ::rig_core::embeddings::EmbeddingModel as _;
use ::rig_core::providers::openai::{CompletionsClient, GenericEmbeddingModel};
use async_trait::async_trait;

use crate::core::traits::knowledge::retrieval::Embedder;
use crate::core::types::knowledge::retrieval::RetrievalError;

/// Embedder over the OpenAI-compatible embeddings model of Rig.
#[derive(Clone)]
pub struct RigEmbedder {
    model: GenericEmbeddingModel<::rig_core::providers::openai::OpenAICompletionsExt>,
    dim: usize,
}

impl RigEmbedder {
    /// Wraps model_name on client. dim must match the index the scraper wrote.
    pub fn new(client: CompletionsClient, model_name: impl Into<String>, dim: usize) -> Self {
        Self {
            model: GenericEmbeddingModel::new(client, model_name, dim),
            dim,
        }
    }
}

#[async_trait]
impl Embedder for RigEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, RetrievalError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let embeddings = self
            .model
            .embed_texts(texts.iter().cloned())
            .await
            .map_err(|e| RetrievalError::Embedding(e.to_string()))?;
        if embeddings.len() != texts.len() {
            return Err(RetrievalError::Embedding(format!(
                "asked for {} vectors, got {}",
                texts.len(),
                embeddings.len()
            )));
        }
        let mut out = Vec::with_capacity(embeddings.len());
        for e in embeddings {
            if e.vec.len() != self.dim {
                return Err(RetrievalError::Embedding(format!(
                    "dimension {} does not match configured {}",
                    e.vec.len(),
                    self.dim
                )));
            }
            // The index stores f32.
            #[allow(clippy::cast_possible_truncation)]
            out.push(e.vec.into_iter().map(|x| x as f32).collect());
        }
        Ok(out)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
