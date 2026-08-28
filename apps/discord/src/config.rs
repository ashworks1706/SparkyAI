//! Bot configuration from `SPARKY_*` env. Only what the bot needs; the API owns everything else.

use figment::{Figment, providers::Env};
use secrecy::SecretString;
use serde::Deserialize;

/// Bot configuration.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Process-level settings.
    pub app: App,
    /// How to reach the API.
    pub api: Api,
    /// Discord credentials and guild.
    pub discord: Discord,
    /// Error reporting.
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

/// How to reach the API.
#[derive(Debug, Deserialize)]
pub struct Api {
    /// Base URL of the api process.
    pub base_url: String,
    /// Shared secret presented on every request.
    pub service_token: SecretString,
}

/// Discord credentials and guild.
#[derive(Debug, Deserialize)]
pub struct Discord {
    /// Bot token.
    pub token: SecretString,
    /// The one guild this deployment serves.
    pub guild_id: u64,
    /// Role name that grants moderator permissions.
    pub mod_role: String,
}

/// Error reporting.
#[derive(Debug, Deserialize)]
pub struct Telemetry {
    /// Sentry DSN; unset disables Sentry.
    pub sentry_dsn: Option<SecretString>,
}

impl Config {
    /// Loads from `SPARKY_*` variables, `__` separating nesting.
    pub fn load() -> anyhow::Result<Self> {
        Ok(Figment::new()
            .merge(Env::prefixed("SPARKY_").split("__"))
            .extract()?)
    }
}
