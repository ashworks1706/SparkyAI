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
    /// The knowledge service: search, memory, conversations.
    pub knowledge: Knowledge,
    /// Sentry and `OpenTelemetry`.
    pub telemetry: Telemetry,
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

/// Chat model served by vLLM (OpenAI-compatible).
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
}

/// The knowledge service.
#[derive(Debug, Deserialize)]
pub struct Knowledge {
    /// Base URL of `knowledge-api`.
    pub base_url: String,
    /// Shared secret presented on every request.
    pub service_token: SecretString,
}

/// Observability sinks. All optional.
#[derive(Debug, Deserialize)]
pub struct Telemetry {
    /// Sentry DSN; unset disables Sentry.
    pub sentry_dsn: Option<SecretString>,
    /// OTLP/gRPC endpoint; unset disables trace export.
    pub otlp_endpoint: Option<String>,
    /// Axiom API token, sent as a bearer header.
    pub axiom_token: Option<SecretString>,
    /// Axiom dataset, sent as `x-axiom-dataset`.
    pub axiom_dataset: Option<String>,
}

impl Config {
    /// Loads from `SPARKY_*` variables, `__` separating nesting: `SPARKY_POSTGRES__URL`.
    pub fn load() -> anyhow::Result<Self> {
        Ok(Figment::new()
            .merge(Env::prefixed("SPARKY_").split("__"))
            .extract()?)
    }
}
