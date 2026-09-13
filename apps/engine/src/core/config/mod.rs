//! Configuration. Every external service and every harness knob is configured here and
//! nowhere else.
//!
//! Two layers, lowest first: an optional TOML file (sparky.toml, or SPARKY_CONFIG_FILE)
//! and SPARKY_* environment variables. Env values win over file values.

pub mod harness;
pub mod http;
pub mod services;

use figment::Figment;
use figment::providers::{Env, Format, Toml};
use serde::Deserialize;

use crate::core::types::agent::AgentConfig;
use crate::core::types::agent::assemble::Budget;

pub use self::harness::*;
pub use self::http::*;
pub use self::services::*;

/// TOML layer read when SPARKY_CONFIG_FILE is unset. Missing is not an error.
pub const DEFAULT_CONFIG_FILE: &str = "sparky.toml";

/// Root configuration, loaded from a TOML file and SPARKY_* environment variables.
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
    /// PostgreSQL, the source of truth and the retrieval index.
    pub postgres: Postgres,
    /// Embedding endpoint, used to embed queries at retrieval time.
    pub embedding: Embedding,
    /// OpenTelemetry export.
    #[serde(default)]
    pub telemetry: Telemetry,
    /// Loop limits and prompt budgets.
    #[serde(default)]
    pub agent: Agent,
    /// The text the harness writes around every prompt.
    #[serde(default)]
    pub prompt: Prompt,
    /// What the risk policy allows, denies, and holds for confirmation.
    #[serde(default)]
    pub policy: Policy,
    /// Hybrid retrieval tuning.
    #[serde(default)]
    pub retrieval: Retrieval,
    /// Which tools are registered.
    #[serde(default)]
    pub tools: Tools,
    /// How a live source query runs.
    #[serde(default)]
    pub query: Query,
    /// How history that no longer fits is compacted.
    #[serde(default)]
    pub compaction: Compaction,
    /// What the guardrail refuses.
    #[serde(default)]
    pub guardrail: Guardrail,
    /// How a sandboxed command runs.
    #[serde(default)]
    pub sandbox: SandboxSettings,
    /// How a turn becomes a profile.
    #[serde(default)]
    pub profile: Profile,
    /// JSONL trace recording.
    #[serde(default)]
    pub trace: Trace,
    /// HTTP surface limits.
    #[serde(default)]
    pub http: Http,
    /// MCP servers exposed as tools.
    #[serde(default)]
    pub mcp: Mcp,
}

/// Default budgets. Every field comes from the agent section.
impl Default for Budget {
    fn default() -> Self {
        Agent::default().budget()
    }
}

/// Default loop settings. Every field comes from the agent section.
impl Default for AgentConfig {
    fn default() -> Self {
        let agent = Agent::default();
        AgentConfig {
            provider_name: Telemetry::default().provider_name.into(),
            model_name: std::sync::Arc::from(""),
            max_steps: agent.max_steps,
            max_model_retries: agent.max_model_retries,
            tool_timeout: std::time::Duration::from_secs(agent.tool_timeout_secs),
            confirmation_ttl: std::time::Duration::from_secs(agent.confirmation_ttl_secs),
            temperature: agent.temperature,
            history_turns: agent.history_turns,
            memory_recall_limit: agent.memory_recall_limit,
            recall_in_public: agent.recall_in_public,
            retry_base_ms: agent.retry_base_ms,
            retry_cap_ms: agent.retry_cap_ms,
            max_span_value_chars: agent.max_span_value_chars,
            retrieval_top_k: Retrieval::default().top_k,
            budget: agent.budget(),
            thinking: agent.thinking,
            // [model] has no defaults; these values are not read from a settings struct.
            max_tokens: 1024,
            usd_per_m_prompt: 0.0,
            usd_per_m_completion: 0.0,
        }
    }
}

/// Why configuration was rejected.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// A layer could not be read or a field could not be parsed.
    #[error("config: {0}")]
    Load(String),
    /// The values parsed but cannot work together.
    #[error("config: {0}")]
    Invalid(String),
}

/// Settings that moved to another section.
///
/// An unknown SPARKY_* variable is not rejected: the apps share one .env, so the engine sees
/// the scraper keys and the scraper sees the engine keys. A variable named here is a boot
/// failure that reports where it went.
const RENAMED: [(&str, &str); 4] = [
    ("SPARKY_AGENT__TRACE_DIR", "SPARKY_TRACE__DIR"),
    ("SPARKY_AGENT__RETRIEVAL_TOP_K", "SPARKY_RETRIEVAL__TOP_K"),
    ("SPARKY_MODEL__THINKING", "SPARKY_AGENT__THINKING__MODE"),
    ("SPARKY_TOOLS__QUERY_SOURCE", "SPARKY_TOOLS__SEARCH"),
];

impl Config {
    /// Fails on a variable that has moved, naming what replaced it. is_set reports whether a
    /// variable is present.
    ///
    /// # Errors
    /// Returns [ConfigError::Invalid] when a renamed variable is still set.
    pub fn reject_renamed(is_set: impl Fn(&str) -> bool) -> Result<(), ConfigError> {
        match RENAMED.into_iter().find(|(old, _)| is_set(old)) {
            Some((old, new)) => Err(ConfigError::Invalid(format!(
                "{old} moved to {new}; set that instead"
            ))),
            None => Ok(()),
        }
    }

    /// Loads the TOML layer then SPARKY_* variables, __ separating nesting:
    /// SPARKY_POSTGRES__URL. Environment values win over the file.
    pub fn load() -> Result<Self, ConfigError> {
        Self::reject_renamed(|key| std::env::var_os(key).is_some())?;
        let path =
            std::env::var("SPARKY_CONFIG_FILE").unwrap_or_else(|_| DEFAULT_CONFIG_FILE.to_owned());
        let cfg: Self = Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("SPARKY_").split("__"))
            .extract()
            .map_err(|e| ConfigError::Load(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Rejects combinations that would leave the engine unable to answer.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let invalid = |m: String| Err(ConfigError::Invalid(m));
        if !self.retrieval.dense && !self.retrieval.lexical {
            return invalid("retrieval.dense and retrieval.lexical are both off".into());
        }
        // Rendering the date in an offset no clock keeps would name the wrong day.
        if !(-14..=14).contains(&self.prompt.utc_offset_hours) {
            return invalid(format!(
                "prompt.utc_offset_hours must be between -14 and 14, got {}",
                self.prompt.utc_offset_hours
            ));
        }
        if self.agent.chars_per_token == 0 {
            return invalid("agent.chars_per_token must be at least 1".into());
        }
        if self.agent.prompt_budget_tokens == 0 {
            return invalid("agent.prompt_budget_tokens must be at least 1".into());
        }
        if !(0.0..=1.0).contains(&self.profile.min_confidence) {
            return invalid("profile.min_confidence must be between 0 and 1".into());
        }
        if self.profile.list_limit == 0 {
            return invalid("profile.list_limit must be at least 1".into());
        }
        if self.profile.request_timeout_secs == 0 {
            return invalid("profile.request_timeout_secs must be at least 1".into());
        }
        validate_thinking(&self.agent.thinking)?;
        if self.agent.max_steps == 0 {
            return invalid("agent.max_steps must be at least 1".into());
        }
        if !(0.0..=1.0).contains(&self.telemetry.sample_ratio) {
            return invalid(format!(
                "telemetry.sample_ratio must be between 0 and 1, got {}",
                self.telemetry.sample_ratio
            ));
        }
        // An empty ai_path is how the AI endpoint is turned off.
        for (name, path) in [
            ("traces_path", &self.telemetry.traces_path),
            ("ai_path", &self.telemetry.ai_path),
        ] {
            if !(path.starts_with('/') || (name == "ai_path" && path.is_empty())) {
                return invalid(format!("telemetry.{name} must start with /, got {path:?}"));
            }
        }
        if self.telemetry.provider_name.trim().is_empty() {
            return invalid("telemetry.provider_name is empty".into());
        }
        if self.retrieval.text_search_config.is_empty()
            || !self
                .retrieval
                .text_search_config
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            return invalid(format!(
                "retrieval.text_search_config must be a plain identifier, got {:?}",
                self.retrieval.text_search_config
            ));
        }
        if self.sandbox.enabled && self.sandbox.runtime.trim().is_empty() {
            return invalid("sandbox.runtime is empty".into());
        }
        if self.compaction.enabled && self.compaction.max_tokens == 0 {
            return invalid("compaction.max_tokens must be at least 1".into());
        }
        if self.tools.search && self.query.poll_ms == 0 {
            return invalid("query.poll_ms must be at least 1".into());
        }
        if self.retrieval.candidates < 1 {
            return invalid("retrieval.candidates must be at least 1".into());
        }
        if self.retrieval.rrf_k < 0.0 {
            return invalid("retrieval.rrf_k must not be negative".into());
        }
        if self.agent.retry_cap_ms < self.agent.retry_base_ms {
            return invalid("agent.retry_cap_ms is below agent.retry_base_ms".into());
        }
        let mut names = std::collections::BTreeSet::new();
        for server in &self.mcp.servers {
            if !names.insert(server.name.clone()) {
                return invalid(format!(
                    "mcp.servers has two servers named {:?}",
                    server.name
                ));
            }
        }
        // A section budget above the total is a configuration error; the prompt is never
        // trimmed.
        for (name, value) in [
            ("evidence", self.agent.evidence_budget_tokens),
            ("history", self.agent.history_budget_tokens),
            ("memory", self.agent.memory_budget_tokens),
            ("capabilities", self.agent.capabilities_budget_tokens),
        ] {
            if value > self.agent.prompt_budget_tokens {
                return invalid(format!(
                    "agent.{name}_budget_tokens ({value}) exceeds agent.prompt_budget_tokens ({})",
                    self.agent.prompt_budget_tokens
                ));
            }
        }
        Ok(())
    }

    /// System instructions: prompt.system_file, else prompt.system, else fallback.
    pub fn system_prompt(&self, fallback: &str) -> Result<String, ConfigError> {
        if let Some(path) = self
            .prompt
            .system_file
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
        {
            return std::fs::read_to_string(path)
                .map(|s| s.trim().to_owned())
                .map_err(|e| ConfigError::Invalid(format!("prompt.system_file {path}: {e}")));
        }
        Ok(self
            .prompt
            .system
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(fallback)
            .to_owned())
    }
}
