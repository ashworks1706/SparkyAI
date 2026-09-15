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
    /// Rows either side of a hit read back with it. 0 hands back the hit alone.
    pub window: i32,
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
            window: 2,
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
    /// How soon after queueing the engine first checks whether the scraper has answered.
    pub poll_ms: u64,
    /// Longest the engine waits between checks. The wait doubles from poll_ms up to this.
    pub poll_max_ms: u64,
    /// Live queries that may reach the database at once, across replicas. Zero removes the cap.
    pub max_in_flight: usize,
    /// How long a query may wait for the scraper to claim it before reporting it not running.
    pub claim_secs: u64,
    /// The cache in front of live queries.
    pub cache: QueryCache,
}

impl Default for Query {
    fn default() -> Self {
        Self {
            timeout_secs: 90,
            poll_ms: 100,
            poll_max_ms: 1_000,
            max_in_flight: 16,
            claim_secs: 5,
            cache: QueryCache::default(),
        }
    }
}

/// Reuse of live query answers, and the lease that keeps one fetch per query.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct QueryCache {
    /// Use the cache. Off fetches every query, every time.
    pub enabled: bool,
    /// How long an answer of a source with no entry in ttl_secs is reused.
    pub default_ttl_secs: u64,
    /// How long an answer of a source the engine offers as search_live_ is reused.
    pub live_ttl_secs: u64,
    /// How long an answer of one source is reused, by registry key. Zero takes handoff_secs.
    pub ttl_secs: std::collections::HashMap<String, u64>,
    /// Floor under every source's lifetime: long enough for the requests that waited to read it.
    pub handoff_secs: u64,
    /// Longest one request holds the lease. Must cover a whole fetch.
    pub lease_secs: u64,
    /// How often a request waiting on the lease looks for the answer.
    pub poll_ms: u64,
    /// Budget for one call to the cache. A slow cache is treated as one that is not there.
    pub timeout_ms: u64,
}

impl Default for QueryCache {
    fn default() -> Self {
        Self {
            enabled: true,
            default_ttl_secs: 900,
            live_ttl_secs: 0,
            ttl_secs: std::collections::HashMap::new(),
            handoff_secs: 5,
            lease_secs: 120,
            poll_ms: 100,
            timeout_ms: 500,
        }
    }
}
