//! Settings for retrieval, the gate in front of it, and live source queries.

use serde::Deserialize;

/// Hybrid retrieval tuning. Dense and lexical are fused with reciprocal rank fusion.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Retrieval {
    /// Evidence chunks handed to the prompt per request.
    pub top_k: usize,
    /// Candidates pulled from each leg before fusion.
    pub candidates: i64,
    /// Reciprocal rank fusion constant. Lower trusts the top of each list more.
    pub rrf_k: f32,
    /// PostgreSQL text search configuration for the lexical leg.
    pub text_search_config: String,
    /// Run the pgvector leg.
    pub dense: bool,
    /// Run the full-text leg.
    pub lexical: bool,
    /// Drop fused results below this score. Zero keeps everything.
    pub min_score: f32,
    /// Cosine distance past which a dense match is dropped. 2.0 keeps everything.
    pub max_distance: f32,
    /// Drop a chunk when the summary covering it is already in the result.
    pub collapse_tree: bool,
    /// The gate retrieval passes before it runs.
    pub router: Router,
}

/// The rule gate on retrieval, run on every turn. Empty lists use the built-in ones.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Router {
    /// Run the gate. Off retrieves for every turn.
    pub enabled: bool,
    /// Longest turn, in words, a chitchat marker may skip retrieval for.
    pub max_chitchat_words: usize,
    /// Markers of small talk.
    pub chitchat: Vec<String>,
    /// Cues that the answer has to be current.
    pub live: Vec<String>,
}

impl Default for Router {
    fn default() -> Self {
        Self {
            enabled: true,
            max_chitchat_words: 4,
            chitchat: Vec::new(),
            live: Vec::new(),
        }
    }
}

impl Default for Retrieval {
    fn default() -> Self {
        Self {
            top_k: 6,
            candidates: 20,
            rrf_k: 60.0,
            text_search_config: "english".into(),
            dense: true,
            lexical: true,
            min_score: 0.0,
            max_distance: 0.6,
            collapse_tree: true,
            router: Router::default(),
        }
    }
}

/// How a live source query runs. The tools section decides whether it is offered at all.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Query {
    /// Budget for one live query, end to end. Overrides agent.tool_timeout_secs for this tool.
    pub timeout_secs: u64,
    /// How often the engine checks whether the scraper has answered.
    pub poll_ms: u64,
    /// How long a query may wait for the scraper to claim it before reporting it not running.
    pub claim_secs: u64,
}

impl Default for Query {
    fn default() -> Self {
        Self {
            timeout_secs: 90,
            poll_ms: 100,
            claim_secs: 5,
        }
    }
}
