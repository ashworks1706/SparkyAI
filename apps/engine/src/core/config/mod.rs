//! Environment configuration. Every external service is configured here and nowhere else.

use figment::{Figment, providers::Env};
use secrecy::SecretString;
use serde::Deserialize;

/// Root configuration, loaded from `SPARKY_*` environment variables.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Process-level settings.
    pub app: App,
    /// How clients reach the engine.
    pub engine: Engine,
    /// Discord bot settings.
    pub discord: Discord,
    /// Chat model endpoint.
    pub model: Model,
    /// `PostgreSQL`, the source of truth and the retrieval index.
    pub postgres: Postgres,
    /// Embedding endpoint, used to embed queries at retrieval time.
    pub embedding: Embedding,
    /// Reranker endpoint, applied to fused retrieval candidates.
    pub reranker: Reranker,
    /// Sentry and `OpenTelemetry`.
    pub telemetry: Telemetry,
    /// Loop limits and prompt budgets.
    pub agent: Agent,
}

/// Process-level settings.
#[derive(Debug, Deserialize)]
pub struct App {
    /// `development`, `staging`, or `production`.
    pub env: String,
    /// Bind address for the HTTP server.
    pub http_addr: String,
    /// `tracing` filter directive, e.g. `info,sparky=debug`.
    pub log_level: String,
}

/// How the Discord bot and other clients reach the engine.
#[derive(Debug, Deserialize)]
pub struct Engine {
    /// Base URL of the engine process.
    pub base_url: String,
    /// Shared secret presented by internal clients.
    pub service_token: SecretString,
}

/// Discord bot settings.
#[derive(Debug, Deserialize)]
pub struct Discord {
    /// Bot token.
    pub token: SecretString,
    /// The one guild this deployment serves.
    pub guild_id: u64,
    /// Role name that grants moderator permissions.
    pub mod_role: String,
}

/// Chat model served by `llama-server` (OpenAI-compatible).
#[derive(Debug, Deserialize)]
pub struct Model {
    /// OpenAI-compatible base URL, ending in `/v1`.
    pub base_url: String,
    /// API key for the endpoint.
    pub api_key: SecretString,
    /// Model name as served.
    pub name: String,
    /// Default completion budget.
    pub max_tokens: u32,
    /// USD per million prompt tokens; zero for local serving.
    #[serde(default)]
    pub usd_per_m_prompt: f64,
    /// USD per million completion tokens; zero for local serving.
    #[serde(default)]
    pub usd_per_m_completion: f64,
}

/// Agent loop limits. Every field has a default so a bare `.env` still boots.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Agent {
    /// Model calls per request.
    pub max_steps: u32,
    /// Retries per model call on transport or 5xx errors.
    pub max_model_retries: u32,
    /// Wall-clock budget per request.
    pub request_timeout_secs: u64,
    /// Budget per tool call.
    pub tool_timeout_secs: u64,
    /// Sampling temperature.
    pub temperature: f32,
    /// Evidence chunks retrieved per request.
    pub retrieval_top_k: usize,
    /// Prior turns loaded into the prompt.
    pub history_turns: usize,
    /// Whole-prompt token budget.
    pub prompt_budget_tokens: usize,
    /// Directory for JSONL traces.
    pub trace_dir: String,
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_model_retries: 2,
            request_timeout_secs: 90,
            tool_timeout_secs: 20,
            temperature: 0.3,
            retrieval_top_k: 6,
            history_turns: 20,
            prompt_budget_tokens: 3_000,
            trace_dir: "traces".into(),
        }
    }
}

/// `PostgreSQL` connection.
#[derive(Debug, Deserialize)]
pub struct Postgres {
    /// `libpq` connection URL.
    pub url: SecretString,
    /// Maximum pooled connections.
    pub max_connections: u32,
}

/// Embedding endpoint (OpenAI-compatible).
#[derive(Debug, Deserialize)]
pub struct Embedding {
    /// Base URL, ending in `/v1`.
    pub base_url: String,
    /// API key for the endpoint.
    pub api_key: SecretString,
    /// Model name as served.
    pub name: String,
    /// Vector dimension; must match the `chunks.embedding` column.
    pub dim: u32,
}

/// Reranker endpoint (OpenAI-compatible `/v1/rerank`).
#[derive(Debug, Deserialize)]
pub struct Reranker {
    /// Base URL, ending in `/v1`.
    pub base_url: String,
    /// API key for the endpoint.
    pub api_key: SecretString,
    /// Model name as served.
    pub name: String,
}

/// Observability sinks. Defaults to the local Phoenix collector; empty endpoint disables export.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Telemetry {
    /// Sentry DSN; unset disables Sentry.
    pub sentry_dsn: Option<SecretString>,
    /// OTLP/gRPC endpoint. Phoenix locally; unset or empty disables trace export.
    pub otlp_endpoint: Option<String>,
    /// Axiom API token, sent as a bearer header.
    pub axiom_token: Option<SecretString>,
    /// Axiom dataset, sent as `x-axiom-dataset`.
    pub axiom_dataset: Option<String>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            sentry_dsn: None,
            otlp_endpoint: Some("http://localhost:4317".into()),
            axiom_token: None,
            axiom_dataset: None,
        }
    }
}

impl Config {
    /// Loads from `SPARKY_*` variables, `__` separating nesting: `SPARKY_POSTGRES__URL`.
    pub fn load() -> anyhow::Result<Self> {
        Ok(Figment::new()
            .merge(Env::prefixed("SPARKY_").split("__"))
            .extract()?)
    }
}
