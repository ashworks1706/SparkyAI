//! `Reranker` over llama-server's `/v1/rerank`.

use std::time::Duration;

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use crate::agent::harness::retrieval::{Reranker, RetrievalError};

/// Rerank client for one model.
#[derive(Debug, Clone)]
pub struct HttpReranker {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
    model: String,
}

impl HttpReranker {
    /// Builds a client. `base_url` ends in `/v1`.
    pub fn new(
        base_url: impl Into<String>,
        api_key: SecretString,
        model: impl Into<String>,
    ) -> Result<Self, RetrievalError> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| RetrievalError::Rerank(e.to_string()))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
            model: model.into(),
        })
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
}

#[derive(Deserialize)]
struct WireResponse {
    results: Vec<WireResult>,
}

#[derive(Deserialize)]
struct WireResult {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl Reranker for HttpReranker {
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>, RetrievalError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let mut request = self
            .http
            .post(format!("{}/rerank", self.base_url))
            .json(&WireRequest {
                model: &self.model,
                query,
                documents,
            });
        if !self.api_key.expose_secret().is_empty() {
            request = request.bearer_auth(self.api_key.expose_secret());
        }
        let response = request
            .send()
            .await
            .map_err(|e| RetrievalError::Rerank(e.to_string()))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| RetrievalError::Rerank(e.to_string()))?;
        if !status.is_success() {
            return Err(RetrievalError::Rerank(format!(
                "{status}: {}",
                text.chars().take(300).collect::<String>()
            )));
        }
        let wire: WireResponse =
            serde_json::from_str(&text).map_err(|e| RetrievalError::Rerank(e.to_string()))?;
        let mut scores = vec![f32::MIN; documents.len()];
        for r in wire.results {
            if let Some(slot) = scores.get_mut(r.index) {
                *slot = r.relevance_score;
            }
        }
        Ok(scores)
    }
}
