//! Structs the adapters are built from: model clients, tools, stores, and route state.
//! Behaviour lives in `agent::model`, `agent::tools`, `stores`, and `routes`.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use uuid::Uuid;

use crate::core::traits::conversation::ConversationStore;
use crate::core::traits::retrieval::{Embedder, Reranker, Retriever};
use crate::core::types::harness::Agent;

/// `ModelProvider` over Rig's chat-completions model.
#[derive(Clone)]
pub struct RigChat {
    pub(crate) model: ::rig_core::providers::openai::CompletionModel,
    pub(crate) name: String,
}

/// `Embedder` over Rig's OpenAI-compatible embeddings model.
#[derive(Clone)]
pub struct RigEmbedder {
    pub(crate) model: ::rig_core::providers::openai::GenericEmbeddingModel<
        ::rig_core::providers::openai::OpenAICompletionsExt,
    >,
    pub(crate) dim: usize,
}

/// Rerank client for one model behind llama-server's `/v1/rerank`.
#[derive(Debug, Clone)]
pub struct HttpReranker {
    pub(crate) http: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: SecretString,
    pub(crate) model: String,
}

/// Wire request for `/v1/rerank`.
#[derive(Serialize)]
pub struct RerankWireRequest<'a> {
    pub(crate) model: &'a str,
    pub(crate) query: &'a str,
    pub(crate) documents: &'a [String],
}

/// Wire response for `/v1/rerank`.
#[derive(Deserialize)]
pub struct RerankWireResponse {
    pub(crate) results: Vec<RerankWireResult>,
}

/// One scored document in a rerank response.
#[derive(Deserialize)]
pub struct RerankWireResult {
    pub(crate) index: usize,
    pub(crate) relevance_score: f32,
}

/// `ReadPublic` search over indexed ASU sources.
pub struct PublicSearch {
    pub(crate) retriever: Arc<dyn Retriever>,
    pub(crate) top_k: usize,
}

/// Arguments the model passes to `search_asu`.
#[derive(Deserialize)]
pub struct SearchArgs {
    pub(crate) query: String,
    #[serde(default)]
    pub(crate) categories: Vec<String>,
}

/// `ExternalWrite`: posts an announcement to a channel. Phase 3 wires it to the bot.
pub struct PostAnnouncement;

/// Hybrid retrieval over the `chunks` table.
pub struct PgRetriever {
    pub(crate) pool: PgPool,
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) reranker: Option<Arc<dyn Reranker>>,
    /// Candidates pulled from each of dense and lexical before fusion.
    pub(crate) candidates: i64,
}

/// One row pulled from `chunks` before fusion.
#[derive(Clone)]
pub struct Candidate {
    pub(crate) chunk_id: Uuid,
    pub(crate) source_id: Uuid,
    pub(crate) title: String,
    pub(crate) url: Option<String>,
    pub(crate) content: String,
    pub(crate) fetched_at: DateTime<Utc>,
}

/// Conversations and messages tables.
pub struct PgConversations {
    pub(crate) pool: PgPool,
}

/// Memories table. Every query is scoped by tenant and user.
pub struct PgMemory {
    pub(crate) pool: PgPool,
}

/// What the chat route needs.
#[derive(Clone)]
pub struct ChatState {
    /// The agent.
    pub agent: Agent,
    /// To create the conversation row before the run.
    pub conversations: Option<Arc<dyn ConversationStore>>,
    /// Per-request wall-clock budget.
    pub request_budget: Duration,
    /// Tenant used when the client sends none (single-guild deployments).
    pub default_tenant: String,
}
