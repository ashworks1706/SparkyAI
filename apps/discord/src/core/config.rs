//! Bot configuration from `SPARKY_*` env. Only what the bot needs; the API owns everything else.

use figment::{Figment, providers::Env};
use secrecy::SecretString;
use serde::Deserialize;

/// Bot configuration.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Process-level settings.
    pub app: App,
    /// How to reach the engine.
    pub engine: Engine,
    /// Discord credentials and guild.
    pub discord: Discord,
    /// Trace export.
    #[serde(default)]
    pub telemetry: Telemetry,
}

/// Process-level settings.
#[derive(Debug, Deserialize)]
pub struct App {
    /// `development`, `staging`, or `production`.
    pub env: String,
    /// `tracing` filter directive.
    pub log_level: String,
}

/// How to reach the engine.
#[derive(Debug, Deserialize)]
pub struct Engine {
    /// Base URL of the engine process.
    pub base_url: String,
    /// Shared secret presented on every request.
    pub service_token: SecretString,
}

/// Discord credentials and guild.
#[derive(Debug, Deserialize)]
pub struct Discord {
    /// Bot token.
    pub token: SecretString,
    /// The one guild this deployment serves. Role checks happen in the engine.
    pub guild_id: u64,
}

/// Trace export. Defaults to the local Phoenix collector; an empty endpoint disables it.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Telemetry {
    /// OTLP/gRPC endpoint.
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
    /// Loads from `SPARKY_*` variables, `__` separating nesting.
    pub fn load() -> anyhow::Result<Self> {
        Ok(Figment::new()
            .merge(Env::prefixed("SPARKY_").split("__"))
            .extract()?)
    }
}
