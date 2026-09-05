//! Environment configuration. Every external service is configured here and nowhere else.

use figment::{Figment, providers::Env};
use secrecy::SecretString;
use serde::Deserialize;

/// Root configuration, loaded from `SPARKY_*` environment variables.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Process-level settings.
    pub app: App,
    /// What clients must present to call the engine.
    pub engine: Engine,
    /// Discord bot settings.
    pub discord: Discord,
    /// Chat model endpoint.
    pub model: Model,
    /// `PostgreSQL`, the source of truth and the retrieval index.
    pub postgres: Postgres,
    /// Embedding endpoint, used to embed queries at retrieval time.
    pub embedding: Embedding,
    /// `OpenTelemetry` export.
    #[serde(default)]
    pub telemetry: Telemetry,
    /// Loop limits and prompt budgets.
    #[serde(default)]
    pub agent: Agent,
    /// MCP servers exposed as tools.
    #[serde(default)]
    pub mcp: Mcp,
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

/// Client authentication.
#[derive(Debug, Deserialize)]
pub struct Engine {
    /// Bearer token every `/chat` caller must present.
    pub service_token: SecretString,
}

/// The guild this engine serves.
#[derive(Debug, Deserialize)]
pub struct Discord {
    /// The one guild this deployment serves.
    pub guild_id: u64,
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
    /// Let Qwen3-style models emit reasoning before answering. Off by default: reasoning is
    /// dropped from the answer and burns the completion budget on small contexts.
    #[serde(default)]
    pub thinking: bool,
}

/// MCP servers exposed as tools. Empty URLs disable a server.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Mcp {
    /// Playwright MCP Streamable-HTTP endpoint, e.g. `http://localhost:8931/mcp`.
    pub playwright_url: Option<String>,
    /// Remote tool names to expose; empty exposes every tool the server lists.
    pub playwright_tools: Vec<String>,
    /// Show the model only each tool's required properties. On by default for small models.
    pub required_props_only: bool,
}

impl Default for Mcp {
    fn default() -> Self {
        Self {
            playwright_url: None,
            // Enough to browse and read; every schema costs context on every step.
            playwright_tools: [
                "browser_navigate",
                "browser_navigate_back",
                "browser_snapshot",
                "browser_click",
                "browser_type",
                "browser_press_key",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            required_props_only: true,
        }
    }
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
    /// Model calls in flight at once. Match `llama-server --parallel`. 0 removes the limit.
    pub model_slots: usize,
    /// How long a request waits for a free model slot before reporting the model busy.
    pub model_queue_wait_secs: u64,
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
            model_slots: 2,
            model_queue_wait_secs: 30,
            temperature: 0.3,
            retrieval_top_k: 6,
            history_turns: 20,
            prompt_budget_tokens: 3_000,
            trace_dir: ".sparky/traces".into(),
        }
    }
}

/// `PostgreSQL` connection.
#[derive(Debug, Deserialize)]
pub struct Postgres {
    /// `libpq` connection URL.
    pub url: SecretString,
    /// Maximum pooled connections.
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
}

fn default_max_connections() -> u32 {
    8
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

/// Trace export. Defaults to the local Phoenix collector; an empty endpoint disables it.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Telemetry {
    /// OTLP/gRPC endpoint. Phoenix locally; unset or empty disables trace export.
    pub otlp_endpoint: Option<String>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            otlp_endpoint: Some("http://localhost:4317".into()),
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
