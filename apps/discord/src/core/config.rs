//! Bot configuration from SPARKY_* env. Only what the bot needs. The API owns everything else.

use figment::Figment;
use figment::providers::{Env, Format, Toml};
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
    /// How the bot behaves in the guild.
    #[serde(default)]
    pub bot: Bot,
}

/// The marker the engine policy reads to allow write-side tools, when none is configured.
pub const WRITE_CAPABILITY: &str = "MANAGE_GUILD";

/// How the bot behaves in the guild. Everything here is presentation and pacing. The engine
/// decides what may run.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Bot {
    /// Channel ids the bot answers in. Empty answers in every channel it can see.
    pub channels: Vec<u64>,
    /// Longest message posted before the reply is split.
    pub max_message_chars: usize,
    /// Shortest gap between edits of the progress message.
    pub edit_every_ms: u64,
    /// Seconds one user must wait between questions. 0 removes the limit.
    pub cooldown_secs: u64,
    /// The role name the engine policy reads to allow write-side tools. Must match
    /// SPARKY_POLICY__WRITE_ROLES on the engine.
    pub write_capability: String,
    /// Minutes of silence before a thread the bot opens archives. One of THREAD_ARCHIVE_MINUTES.
    pub thread_auto_archive_minutes: u16,
}

/// Smallest bot.max_message_chars accepted.
pub const MIN_MESSAGE_CHARS: usize = 200;

/// Auto archive durations Discord accepts, in minutes.
pub const THREAD_ARCHIVE_MINUTES: [u16; 4] = [60, 1_440, 4_320, 10_080];

impl Default for Bot {
    fn default() -> Self {
        Self {
            channels: Vec::new(),
            max_message_chars: 2_000,
            edit_every_ms: 1_500,
            cooldown_secs: 0,
            write_capability: WRITE_CAPABILITY.to_owned(),
            thread_auto_archive_minutes: 1_440,
        }
    }
}

/// Process-level settings.
#[derive(Debug, Deserialize)]
pub struct App {
    /// One of development, staging, or production.
    pub env: String,
    /// The tracing filter directive.
    pub log_level: String,
}

/// How to reach the engine.
#[derive(Debug, Deserialize)]
pub struct Engine {
    /// Base URL of the engine process.
    pub base_url: String,
    /// Shared secret presented on every request.
    pub service_token: SecretString,
    /// How long to wait for the connection.
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    /// How long to wait for a whole answer. Keep it above the engine request budget.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
}

fn default_connect_timeout_secs() -> u64 {
    5
}

fn default_request_timeout_secs() -> u64 {
    120
}

/// Discord credentials and guild.
#[derive(Debug, Deserialize)]
pub struct Discord {
    /// Bot token.
    pub token: SecretString,
    /// The one guild this deployment serves. Role checks happen in the engine.
    pub guild_id: u64,
}

/// Trace export. Defaults to the local Phoenix collector. An empty endpoint disables it.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Telemetry {
    /// OTLP/gRPC endpoint.
    pub otlp_endpoint: Option<String>,
    /// The service.name on exported spans. Defaults to discord.
    pub service_name: Option<String>,
    /// Fraction of traces exported, 0.0 to 1.0.
    pub sample_ratio: f64,
    /// Budget for one export batch.
    pub export_timeout_secs: u64,
    /// Only spans whose target starts with this are exported. Defaults to discord.
    pub span_target_prefix: Option<String>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            otlp_endpoint: Some("http://localhost:4317".into()),
            service_name: None,
            sample_ratio: 1.0,
            export_timeout_secs: 10,
            span_target_prefix: None,
        }
    }
}

/// TOML layer read when SPARKY_CONFIG_FILE is unset. Missing is not an error.
pub const DEFAULT_CONFIG_FILE: &str = "sparky.toml";

impl Config {
    /// Loads the TOML layer then SPARKY_* variables, with __ separating nesting. Environment
    /// values win.
    pub fn load() -> anyhow::Result<Self> {
        let path =
            std::env::var("SPARKY_CONFIG_FILE").unwrap_or_else(|_| DEFAULT_CONFIG_FILE.to_owned());
        let cfg: Self = Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("SPARKY_").split("__"))
            .extract()?;
        if !(0.0..=1.0).contains(&cfg.telemetry.sample_ratio) {
            anyhow::bail!(
                "telemetry.sample_ratio must be between 0 and 1, got {}",
                cfg.telemetry.sample_ratio
            );
        }
        if !(MIN_MESSAGE_CHARS..=2_000).contains(&cfg.bot.max_message_chars) {
            anyhow::bail!(
                "bot.max_message_chars must be between {MIN_MESSAGE_CHARS} and 2000, got {}",
                cfg.bot.max_message_chars
            );
        }
        if !THREAD_ARCHIVE_MINUTES.contains(&cfg.bot.thread_auto_archive_minutes) {
            anyhow::bail!(
                "bot.thread_auto_archive_minutes must be one of {THREAD_ARCHIVE_MINUTES:?}, got {}",
                cfg.bot.thread_auto_archive_minutes
            );
        }
        Ok(cfg)
    }
}
