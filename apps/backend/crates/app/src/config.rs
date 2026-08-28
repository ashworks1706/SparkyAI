//! Environment configuration. Every external service is configured here and nowhere else.
//!
//! Fields are read by adapters as they are wired in (Phases 1–3).
#![allow(dead_code)]

use figment::{Figment, providers::Env};
use secrecy::SecretString;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub app: App,
    pub api: Api,
    pub discord: Discord,
    pub model: Model,
    pub embedding: Embedding,
    pub reranker: Reranker,
    pub postgres: Postgres,
    pub redis: Redis,
    pub qdrant: Qdrant,
    pub object_store: ObjectStore,
    pub telemetry: Telemetry,
}

#[derive(Debug, Deserialize)]
pub struct App {
    pub env: String,
    pub http_addr: String,
    pub log_level: String,
}

/// How the Discord bot (and other clients) reach the API.
#[derive(Debug, Deserialize)]
pub struct Api {
    pub base_url: String,
    pub service_token: SecretString,
}

#[derive(Debug, Deserialize)]
pub struct Discord {
    pub token: SecretString,
    pub guild_id: u64,
    pub mod_role: String,
}

/// Chat model served by vLLM (OpenAI-compatible).
#[derive(Debug, Deserialize)]
pub struct Model {
    pub base_url: String,
    pub api_key: SecretString,
    pub name: String,
    pub max_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct Embedding {
    pub base_url: String,
    pub api_key: SecretString,
    pub name: String,
    pub dim: u64,
}

#[derive(Debug, Deserialize)]
pub struct Reranker {
    pub base_url: String,
    pub api_key: SecretString,
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct Postgres {
    pub url: SecretString,
    pub max_connections: u32,
}

#[derive(Debug, Deserialize)]
pub struct Redis {
    pub url: SecretString,
}

#[derive(Debug, Deserialize)]
pub struct Qdrant {
    pub url: String,
    pub api_key: Option<SecretString>,
    pub collection: String,
}

#[derive(Debug, Deserialize)]
pub struct ObjectStore {
    pub endpoint: String,
    pub bucket: String,
    pub access_key: SecretString,
    pub secret_key: SecretString,
}

#[derive(Debug, Deserialize)]
pub struct Telemetry {
    pub sentry_dsn: Option<SecretString>,
    pub otlp_endpoint: Option<String>,
    pub axiom_token: Option<SecretString>,
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
