//! Wire structs for adapters that speak raw HTTP, and tool argument schemas.

use serde::{Deserialize, Serialize};

/// Request body for llama-server's `/v1/rerank`.
#[derive(Serialize)]
pub struct RerankRequest<'a> {
    /// Model name as served.
    pub model: &'a str,
    /// The query documents are scored against.
    pub query: &'a str,
    /// Documents to score.
    pub documents: &'a [String],
}

/// Response body for `/v1/rerank`.
#[derive(Deserialize)]
pub struct RerankResponse {
    /// One entry per scored document.
    pub results: Vec<RerankResult>,
}

/// One scored document.
#[derive(Deserialize)]
pub struct RerankResult {
    /// Index into the request's `documents`.
    pub index: usize,
    /// Uncalibrated score; ordering only.
    pub relevance_score: f32,
}

/// Arguments the model passes to `search_asu`.
#[derive(Deserialize)]
pub struct SearchArgs {
    /// What to look for.
    pub query: String,
    /// Optional source categories to restrict to.
    #[serde(default)]
    pub categories: Vec<String>,
}
