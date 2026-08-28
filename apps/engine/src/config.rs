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
    /// Embedding model endpoint.
    pub embedding: Embedding,
    /// Reranker endpoint.
    pub reranker: Reranker,
    /// `PostgreSQL`.
    pub postgres: Postgres,
    /// Redis.
    pub redis: Redis,
    /// Qdrant.
    pub qdrant: Qdrant,
    /// S3-compatible object storage.
    pub object_store: ObjectStore,
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

/// Embedding model served by vLLM.
#[derive(Debug, Deserialize)]
pub struct Embedding {
    /// OpenAI-compatible base URL.
    pub base_url: String,
    /// API key for the endpoint.
    pub api_key: SecretString,
    /// Model name as served.
    pub name: String,
    /// Vector dimension; must match the `Qdrant` collection.
    pub dim: u64,
}

/// Reranker served by vLLM.
#[derive(Debug, Deserialize)]
pub struct Reranker {
    /// Base URL.
    pub base_url: String,
    /// API key for the endpoint.
    pub api_key: SecretString,
    /// Model name as served.
    pub name: String,
}

/// `PostgreSQL` connection.
#[derive(Debug, Deserialize)]
pub struct Postgres {
    /// Connection URL.
    pub url: SecretString,
    /// Pool size.
    pub max_connections: u32,
}

/// Redis connection.
#[derive(Debug, Deserialize)]
pub struct Redis {
    /// Connection URL.
    pub url: SecretString,
}

/// Qdrant connection.
#[derive(Debug, Deserialize)]
pub struct Qdrant {
    /// gRPC URL.
    pub url: String,
    /// API key, if the instance requires one.
    pub api_key: Option<SecretString>,
    /// Collection holding document chunks.
    pub collection: String,
}

/// S3-compatible object storage.
#[derive(Debug, Deserialize)]
pub struct ObjectStore {
    /// Endpoint URL.
    pub endpoint: String,
    /// Bucket name.
    pub bucket: String,
    /// Access key.
    pub access_key: SecretString,
    /// Secret key.
    pub secret_key: SecretString,
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
