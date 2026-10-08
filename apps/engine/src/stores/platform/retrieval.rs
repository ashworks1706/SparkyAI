//! Knowledge search on the platform: the organization's sources plus public ones, fused by the platform.

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use super::client::{Call, PlatformClient, field, parse_time};
use crate::core::traits::knowledge::retrieval::{Embedder, Retriever};
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::knowledge::retrieval::{RetrievalError, RetrievalQuery};

/// Namespace of the ids derived from platform source keys and chunk ids.
const SOURCE_NS: Uuid = Uuid::from_u128(0x5f0c_5a2e_8f3b_4d61_9c1e_7a2b_3c4d_5e6f);

/// What a platform search sends besides the query. Built from the platform and retrieval sections.
#[derive(Debug, Clone)]
pub struct SearchTuning {
    /// Model name sent with the query vector.
    pub embedding_model: String,
    /// Neighbor chunks either side of a hit read back with it.
    pub window: i32,
    /// Longest query text sent, in characters.
    pub max_query_chars: usize,
}

/// Knowledge search over the platform knowledge module.
pub struct PlatformRetriever {
    client: PlatformClient,
    embedder: Option<Arc<dyn Embedder>>,
    tuning: SearchTuning,
}

impl PlatformRetriever {
    /// Builds a retriever. Without an embedder the platform searches as it is configured to.
    pub fn new(
        client: PlatformClient,
        embedder: Option<Arc<dyn Embedder>>,
        tuning: SearchTuning,
    ) -> Self {
        Self {
            client,
            embedder,
            tuning,
        }
    }

    /// The query vector, or None when there is no embedder or it failed.
    async fn vector(&self, text: &str) -> Option<Vec<f32>> {
        let embedder = self.embedder.as_ref()?;
        match embedder.embed(&[text.to_owned()]).await {
            Ok(mut vectors) => vectors.pop(),
            Err(error) => {
                tracing::warn!(%error, "query embedding failed; searching the platform on text only");
                None
            }
        }
    }
}

/// An id stable across calls: the value itself when it is a UUID, else one derived from it.
fn stable_id(value: &str) -> Uuid {
    Uuid::parse_str(value).unwrap_or_else(|_| Uuid::new_v5(&SOURCE_NS, value.as_bytes()))
}

/// One result row as evidence, or None when it has no fetch time.
fn evidence(row: &Value) -> Result<Option<Evidence>, RetrievalError> {
    let read = |e: super::client::PlatformError| RetrievalError::Store(e.to_string());
    let key: String = field(row, "source_key").map_err(read)?;
    let chunk: String = field(row, "chunk_id").map_err(read)?;
    let title: Option<String> = field(row, "title").map_err(read)?;
    let Some(fetched_at) = row
        .get("fetched_at")
        .and_then(Value::as_str)
        .and_then(parse_time)
    else {
        tracing::warn!(source = %key, "platform result has no fetch time; dropped");
        return Ok(None);
    };
    let score: f64 = field(row, "score").map_err(read)?;
    // A fused score is a small positive fraction.
    #[allow(clippy::cast_possible_truncation)]
    Ok(Some(Evidence {
        source_id: Uuid::new_v5(&SOURCE_NS, key.as_bytes()),
        chunk_id: stable_id(&chunk),
        title: title
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| key.clone()),
        key,
        content: field(row, "content").map_err(read)?,
        url: field(row, "url").map_err(read)?,
        fetched_at,
        score: score as f32,
    }))
}

#[async_trait]
impl Retriever for PlatformRetriever {
    async fn retrieve(
        &self,
        _ctx: &RequestContext,
        query: &RetrievalQuery,
    ) -> Result<Vec<Evidence>, RetrievalError> {
        let text: String = query
            .text
            .chars()
            .take(self.tuning.max_query_chars)
            .collect();
        let mut body = Map::new();
        body.insert("query".into(), json!(text));
        body.insert("top_k".into(), json!(query.top_k));
        body.insert("window".into(), json!(self.tuning.window));
        if let Some(category) = &query.category {
            body.insert("category".into(), json!(category));
        }
        if let Some(vector) = self.vector(&text).await {
            body.insert("embedding".into(), json!(vector));
            body.insert("embedding_model".into(), json!(self.tuning.embedding_model));
        }
        let body = Value::Object(body);
        let value = self
            .client
            .call(&Call::new(Method::POST, "/api/knowledge/search").body(&body))
            .await
            .map_err(|e| RetrievalError::Store(e.to_string()))?;
        let rows: Vec<Value> =
            field(&value, "results").map_err(|e| RetrievalError::Store(e.to_string()))?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            if let Some(found) = evidence(row)? {
                out.push(found);
            }
        }
        Ok(out)
    }
}
