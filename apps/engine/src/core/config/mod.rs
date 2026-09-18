//! Configuration. Every external service and harness knob is configured here, nowhere else.

pub mod harness;
pub mod http;
pub mod services;

use figment::Figment;
use figment::providers::{Env, Format, Toml};
use secrecy::ExposeSecret;
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
    /// Redis, the shared cache in front of live queries. Absent turns query.cache off.
    #[serde(default)]
    pub redis: Option<Redis>,
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
            tool_result_to_file_chars: agent.tool_result_to_file_chars,
            confirmation_ttl: std::time::Duration::from_secs(agent.confirmation_ttl_secs),
            temperature: agent.temperature,
            history_turns: agent.history_turns,
            history_keep: Compaction::default().keep_tokens(agent.history_budget_tokens),
            memory_recall_limit: agent.memory_recall_limit,
            recall_in_public: agent.recall_in_public,
            retry_base_ms: agent.retry_base_ms,
            retry_cap_ms: agent.retry_cap_ms,
            max_span_value_chars: agent.max_span_value_chars,
            retrieval_top_k: Retrieval::default().top_k,
            budget: agent.budget(),
            stream: agent.stream,
            stream_block_chars: agent.stream_block_chars,
            thinking: agent.thinking,
            // The model section has no defaults; these are fixed values.
            max_tokens: 1024,
            max_tokens_without_thinking: 1024,
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

/// Settings that moved to another section. A variable named here fails boot.
const RENAMED: [(&str, &str); 4] = [
    ("SPARKY_AGENT__TRACE_DIR", "SPARKY_TRACE__DIR"),
    ("SPARKY_AGENT__RETRIEVAL_TOP_K", "SPARKY_RETRIEVAL__TOP_K"),
    ("SPARKY_MODEL__THINKING", "SPARKY_AGENT__THINKING__MODE"),
    ("SPARKY_TOOLS__QUERY_SOURCE", "SPARKY_TOOLS__SEARCH"),
];

impl Config {
    /// Fails on a variable that has moved, naming what replaced it.
    pub fn reject_renamed(is_set: impl Fn(&str) -> bool) -> Result<(), ConfigError> {
        match RENAMED.into_iter().find(|(old, _)| is_set(old)) {
            Some((old, new)) => Err(ConfigError::Invalid(format!(
                "{old} moved to {new}; set that instead"
            ))),
            None => Ok(()),
        }
    }

    /// Loads the TOML layer then SPARKY_* variables, __ separating nesting: SPARKY_POSTGRES__URL.
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
        if !(-14..=14).contains(&self.prompt.utc_offset_hours) {
            return invalid(format!(
                "prompt.utc_offset_hours must be between -14 and 14, got {}",
                self.prompt.utc_offset_hours
            ));
        }
        validate_agent(&self.agent)?;
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
        validate_telemetry(&self.telemetry)?;
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
        validate_sandbox(&self.sandbox)?;
        validate_model(&self.model)?;
        if self.compaction.enabled && self.compaction.max_tokens == 0 {
            return invalid("compaction.max_tokens must be at least 1".into());
        }
        validate_compaction(&self.compaction, self.agent.history_budget_tokens)?;
        if self.tools.search && (self.query.poll_ms == 0 || self.query.claim_secs == 0) {
            return invalid("query.poll_ms and query.claim_secs must be at least 1".into());
        }
        if self.tools.search && self.query.poll_max_ms < self.query.poll_ms {
            return invalid(format!(
                "query.poll_max_ms ({}) is below query.poll_ms ({})",
                self.query.poll_max_ms, self.query.poll_ms
            ));
        }
        validate_tools(&self.tools)?;
        validate_query_cache(self)?;
        if self.retrieval.candidates < 1 {
            return invalid("retrieval.candidates must be at least 1".into());
        }
        if !(self.retrieval.max_distance > 0.0 && self.retrieval.max_distance <= 2.0) {
            return invalid(format!(
                "retrieval.max_distance must be above 0 and at most 2, got {}",
                self.retrieval.max_distance
            ));
        }
        if self.retrieval.rrf_k < 0.0 {
            return invalid("retrieval.rrf_k must not be negative".into());
        }
        if self.retrieval.window < 0 {
            return invalid(format!(
                "retrieval.window must not be negative, got {}",
                self.retrieval.window
            ));
        }
        if self.agent.retry_cap_ms < self.agent.retry_base_ms {
            return invalid("agent.retry_cap_ms is below agent.retry_base_ms".into());
        }
        let mut names = std::collections::BTreeSet::new();
        for server in &self.mcp.servers {
            if !names.insert(server.name.as_str()) {
                return invalid(format!(
                    "mcp.servers has two servers named {:?}",
                    server.name
                ));
            }
        }
        // Each section budget fits within the total.
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

/// Rejects search wording the model would read as an empty string, and a missing fallback source.
fn validate_compaction(compaction: &Compaction, history: usize) -> Result<(), ConfigError> {
    if !compaction.enabled {
        return Ok(());
    }
    if !(compaction.keep_share > 0.0 && compaction.keep_share < 1.0) {
        return Err(ConfigError::Invalid(
            "compaction.keep_share must be above 0 and below 1".into(),
        ));
    }
    let needed = compaction.keep_tokens(history) + compaction.max_tokens as usize;
    if needed > history {
        return Err(ConfigError::Invalid(format!(
            "a compaction keeps {} tokens of turns and writes a summary of up to {} \
             (compaction.keep_share and compaction.max_tokens), {needed} in all, over \
             agent.history_budget_tokens = {history}; the summary would be trimmed from the prompt",
            compaction.keep_tokens(history),
            compaction.max_tokens
        )));
    }
    Ok(())
}

fn validate_sandbox(sandbox: &SandboxSettings) -> Result<(), ConfigError> {
    if !sandbox.enabled {
        return Ok(());
    }
    if sandbox.runtime.trim().is_empty() {
        return Err(ConfigError::Invalid("sandbox.runtime is empty".into()));
    }
    if sandbox.egress {
        for (key, value) in [
            ("sandbox.egress_network", &sandbox.egress_network),
            ("sandbox.egress_proxy_image", &sandbox.egress_proxy_image),
            ("sandbox.egress_proxy_name", &sandbox.egress_proxy_name),
        ] {
            if value.trim().is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "{key} is empty and sandbox.egress is on"
                )));
            }
        }
    }
    Ok(())
}

fn validate_tools(tools: &Tools) -> Result<(), ConfigError> {
    if !tools.search {
        return Ok(());
    }
    if tools.live_default_source.trim().is_empty() {
        return Err(ConfigError::Invalid(
            "tools.live_default_source is empty; name the source search_live falls back to".into(),
        ));
    }
    for (name, text) in [
        ("knowledge_description", &tools.knowledge_description),
        ("live_description", &tools.live_description),
        ("query_description", &tools.query_description),
        ("source_description", &tools.source_description),
        ("nothing_stored", &tools.nothing_stored),
    ] {
        if text.trim().is_empty() {
            return Err(ConfigError::Invalid(format!(
                "tools.{name} is empty; the model would be given no wording for the search tools"
            )));
        }
    }
    Ok(())
}

/// Rejects a query cache that cannot keep one fetch per query, or that has nowhere to keep it.
fn validate_query_cache(cfg: &Config) -> Result<(), ConfigError> {
    let invalid = |m: String| Err(ConfigError::Invalid(m));
    let cache = &cfg.query.cache;
    if !(cfg.tools.search && cache.enabled) {
        return Ok(());
    }
    let Some(redis) = &cfg.redis else {
        return invalid(
            "query.cache.enabled needs a redis section; set SPARKY_REDIS__URL or turn it off"
                .into(),
        );
    };
    if let Err(error) = redis::Client::open(redis.url.expose_secret()) {
        return invalid(format!(
            "SPARKY_REDIS__URL is not a Redis URL ({error}); it takes the form \
             redis://host:port, for example redis://localhost:6379"
        ));
    }
    // A lease that expires mid fetch lets a second request fetch the same query.
    if cache.lease_secs < cfg.query.timeout_secs {
        return invalid(format!(
            "query.cache.lease_secs ({}) is below query.timeout_secs ({}), so a lease can expire \
             while its fetch is still running",
            cache.lease_secs, cfg.query.timeout_secs
        ));
    }
    if cache.handoff_secs == 0 {
        return invalid(
            "query.cache.handoff_secs must be at least 1, or a request that waited reads nothing"
                .into(),
        );
    }
    if cache.poll_ms == 0 || cache.timeout_ms == 0 {
        return invalid("query.cache.poll_ms and query.cache.timeout_ms must be at least 1".into());
    }
    Ok(())
}

/// Rejects telemetry settings that cannot export.
fn validate_telemetry(telemetry: &Telemetry) -> Result<(), ConfigError> {
    let invalid = |m: String| Err(ConfigError::Invalid(m));
    if !(0.0..=1.0).contains(&telemetry.sample_ratio) {
        return invalid(format!(
            "telemetry.sample_ratio must be between 0 and 1, got {}",
            telemetry.sample_ratio
        ));
    }
    if telemetry.provider_name.trim().is_empty() {
        return invalid("telemetry.provider_name is empty".into());
    }
    if telemetry.project_name.trim().is_empty() {
        return invalid("telemetry.project_name is empty".into());
    }
    Ok(())
}
