//! `Embedder` over `/v1/embeddings`.

use std::time::Duration;

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use crate::agent::harness::retrieval::{Embedder, RetrievalError};

/// Embedding client for one model.
#[derive(Debug, Clone)]
pub struct HttpEmbedder {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
    model: String,
    dim: usize,
}

impl HttpEmbedder {
    /// Builds a client. `dim` must match the index the scraper wrote.
    pub fn new(
        base_url: impl Into<String>,
        api_key: SecretString,
        model: impl Into<String>,
        dim: usize,
    ) -> Result<Self, RetrievalError> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| RetrievalError::Embedding(e.to_string()))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
            model: model.into(),
            dim,
        })
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct WireResponse {
    data: Vec<WireItem>,
}

#[derive(Deserialize)]
struct WireItem {
    index: usize,
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for HttpEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, RetrievalError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut request = self
            .http
            .post(format!("{}/embeddings", self.base_url))
            .json(&WireRequest {
                model: &self.model,
                input: texts,
            });
        if !self.api_key.expose_secret().is_empty() {
            request = request.bearer_auth(self.api_key.expose_secret());
        }
        let response = request
            .send()
            .await
            .map_err(|e| RetrievalError::Embedding(e.to_string()))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| RetrievalError::Embedding(e.to_string()))?;
        if !status.is_success() {
            return Err(RetrievalError::Embedding(format!(
                "{status}: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        let wire: WireResponse =
            serde_json::from_str(&text).map_err(|e| RetrievalError::Embedding(e.to_string()))?;
        let mut items = wire.data;
        items.sort_by_key(|i| i.index);
        let vectors: Vec<Vec<f32>> = items.into_iter().map(|i| i.embedding).collect();
        if vectors.len() != texts.len() {
            return Err(RetrievalError::Embedding(format!(
                "asked for {} vectors, got {}",
                texts.len(),
                vectors.len()
            )));
        }
        if let Some(bad) = vectors.iter().find(|v| v.len() != self.dim) {
            return Err(RetrievalError::Embedding(format!(
                "dimension {} does not match configured {}",
                bad.len(),
                self.dim
            )));
        }
        Ok(vectors)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
