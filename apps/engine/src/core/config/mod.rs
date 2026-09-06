//! Configuration. Every external service and every harness knob is configured here and
//! nowhere else.
//!
//! Two layers, lowest first: an optional TOML file (`sparky.toml`, or `SPARKY_CONFIG_FILE`)
//! and `SPARKY_*` environment variables. Env always wins, so secrets stay out of the file and
//! lists and nested tables — MCP servers above all — stay out of the environment.

use figment::Figment;
use figment::providers::{Env, Format, Toml};
use secrecy::SecretString;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::core::types::tool::RiskClass;

/// TOML layer read when `SPARKY_CONFIG_FILE` is unset. Missing is not an error.
pub const DEFAULT_CONFIG_FILE: &str = "sparky.toml";

/// Root configuration, loaded from a TOML file and `SPARKY_*` environment variables.
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
    /// The text the harness writes around every prompt.
    #[serde(default)]
    pub prompt: Prompt,
    /// What the risk policy allows, denies, and holds for confirmation.
    #[serde(default)]
    pub policy: PolicyRules,
    /// Hybrid retrieval tuning.
    #[serde(default)]
    pub retrieval: Retrieval,
    /// Which tools are registered.
    #[serde(default)]
    pub tools: Tools,
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

/// Provider sampling parameters. Every field is optional; only what is set is sent, so the
/// server's own defaults apply to the rest.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Sampling {
    /// Nucleus sampling cutoff.
    pub top_p: Option<f64>,
    /// Keep only the k most likely tokens.
    pub top_k: Option<u32>,
    /// Drop tokens below this fraction of the top token's probability.
    pub min_p: Option<f64>,
    /// Penalty applied to tokens already in the context.
    pub repeat_penalty: Option<f64>,
    /// `OpenAI`-style presence penalty.
    pub presence_penalty: Option<f64>,
    /// `OpenAI`-style frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Fixes sampling, so a run reproduces. Set it for eval.
    pub seed: Option<u64>,
    /// Stop sequences.
    pub stop: Vec<String>,
}

impl Sampling {
    /// The set fields as provider request keys.
    pub fn to_json(&self) -> Map<String, Value> {
        let mut map = Map::new();
        let mut put = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                map.insert(key.to_owned(), value);
            }
        };
        put(
            "top_p",
            self.top_p
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number),
        );
        put("top_k", self.top_k.map(|v| Value::Number(v.into())));
        put(
            "min_p",
            self.min_p
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number),
        );
        put(
            "repeat_penalty",
            self.repeat_penalty
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number),
        );
        put(
            "presence_penalty",
            self.presence_penalty
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number),
        );
        put(
            "frequency_penalty",
            self.frequency_penalty
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number),
        );
        put("seed", self.seed.map(|v| Value::Number(v.into())));
        if !self.stop.is_empty() {
            map.insert(
                "stop".to_owned(),
                Value::Array(self.stop.iter().map(|s| Value::String(s.clone())).collect()),
            );
        }
        map
    }
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
    /// Sampling parameters sent with every completion.
    #[serde(default)]
    pub sampling: Sampling,
    /// A JSON object merged into the provider request, for anything `sampling` does not name.
    /// Set keys win over `sampling`.
    #[serde(default)]
    pub extra_params_json: Option<String>,
}

impl Model {
    /// Provider-specific request fields: the chat template switch, `sampling`, then
    /// `extra_params_json` on top.
    pub fn additional_params(&self) -> Result<Value, ConfigError> {
        let mut map = self.sampling.to_json();
        map.insert(
            "chat_template_kwargs".to_owned(),
            serde_json::json!({ "enable_thinking": self.thinking }),
        );
        if let Some(raw) = self
            .extra_params_json
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let Value::Object(extra) = serde_json::from_str(raw)
                .map_err(|e| ConfigError::Invalid(format!("model.extra_params_json: {e}")))?
            else {
                return Err(ConfigError::Invalid(
                    "model.extra_params_json must be a JSON object".into(),
                ));
            };
            map.extend(extra);
        }
        Ok(Value::Object(map))
    }
}

/// MCP servers exposed as tools.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Mcp {
    /// Servers to connect to at boot. Expressed in the TOML layer; the environment cannot
    /// carry a list of tables.
    pub servers: Vec<McpServer>,
    /// Default for a server that does not set `required_props_only`.
    pub required_props_only: bool,
    /// Longest tool result handed back to the model; page snapshots can be enormous.
    pub max_output_chars: usize,
    /// Longest per-property description kept in a tool schema.
    pub max_schema_description_chars: usize,
    /// Longest tool description kept.
    pub max_tool_description_chars: usize,
    /// Legacy single-server form, folded into `servers` as `playwright`. Prefer `servers`.
    pub playwright_url: Option<String>,
    /// Tools exposed by the legacy `playwright_url` server.
    pub playwright_tools: Vec<String>,
}

/// One MCP server.
#[derive(Debug, Clone, Deserialize)]
pub struct McpServer {
    /// Name used in logs and errors.
    pub name: String,
    /// Streamable-HTTP endpoint, e.g. `http://localhost:8931/mcp`.
    pub url: String,
    /// Remote tool names to expose; empty exposes every tool the server lists.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Overrides `mcp.required_props_only` for this server.
    #[serde(default)]
    pub required_props_only: Option<bool>,
    /// Overrides `agent.tool_timeout_secs` for this server's tools.
    #[serde(default)]
    pub tool_timeout_secs: Option<u64>,
}

impl Mcp {
    /// Configured servers, with the legacy `playwright_url` folded in and empty URLs dropped.
    pub fn resolved_servers(&self) -> Vec<McpServer> {
        let mut out: Vec<McpServer> = self
            .servers
            .iter()
            .filter(|s| !s.url.trim().is_empty())
            .cloned()
            .collect();
        if let Some(url) = self
            .playwright_url
            .as_deref()
            .map(str::trim)
            .filter(|u| !u.is_empty())
            && !out.iter().any(|s| s.name == "playwright")
        {
            out.push(McpServer {
                name: "playwright".into(),
                url: url.to_owned(),
                tools: self.playwright_tools.clone(),
                required_props_only: None,
                tool_timeout_secs: None,
            });
        }
        out
    }
}

impl Default for Mcp {
    fn default() -> Self {
        Self {
            servers: Vec::new(),
            required_props_only: true,
            max_output_chars: 6_000,
            max_schema_description_chars: 80,
            max_tool_description_chars: 160,
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
    /// Budget per tool call, unless the tool declares its own.
    pub tool_timeout_secs: u64,
    /// How long a held action waits for its caller's approval.
    pub confirmation_ttl_secs: u64,
    /// Model calls in flight at once. Match `llama-server --parallel`. 0 removes the limit.
    pub model_slots: usize,
    /// How long a request waits for a free model slot before reporting the model busy.
    pub model_queue_wait_secs: u64,
    /// Sampling temperature.
    pub temperature: f32,
    /// Prior turns loaded into the prompt.
    pub history_turns: usize,
    /// Memories recalled per request.
    pub memory_recall_limit: usize,
    /// Whole-prompt token budget.
    pub prompt_budget_tokens: usize,
    /// Cap on the evidence section.
    pub evidence_budget_tokens: usize,
    /// Cap on prior turns.
    pub history_budget_tokens: usize,
    /// Cap on the memory section.
    pub memory_budget_tokens: usize,
    /// Characters per token the budget estimator assumes. Lower it for a tokenizer that
    /// splits ASU jargon and URLs finely.
    pub chars_per_token: usize,
    /// First retry wait, doubled per attempt.
    pub retry_base_ms: u64,
    /// Longest retry wait.
    pub retry_cap_ms: u64,
    /// Longest value recorded on a span; the JSONL trace keeps the rest.
    pub max_span_value_chars: usize,
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_model_retries: 2,
            request_timeout_secs: 90,
            tool_timeout_secs: 20,
            confirmation_ttl_secs: 600,
            model_slots: 2,
            model_queue_wait_secs: 30,
            temperature: 0.3,
            history_turns: 20,
            memory_recall_limit: 10,
            prompt_budget_tokens: 3_000,
            evidence_budget_tokens: 1_200,
            history_budget_tokens: 1_000,
            memory_budget_tokens: 300,
            chars_per_token: 4,
            retry_base_ms: 250,
            retry_cap_ms: 8_000,
            max_span_value_chars: 32_000,
        }
    }
}

/// The text the harness writes around every prompt. Changing any of it changes the prompt
/// hash, so a trace says which wording produced an answer.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Prompt {
    /// System instructions. Overrides the built-in default; overridden by `system_file`.
    pub system: Option<String>,
    /// Path to a file holding the system instructions. Read once at boot.
    pub system_file: Option<String>,
    /// Line naming the user, with `{user}` and `{roles}`.
    pub role_line: String,
    /// Line naming a user who holds no roles, with `{user}`.
    pub role_line_no_roles: String,
    /// Heading above recalled memories.
    pub memory_header: String,
    /// Heading above retrieved evidence.
    pub evidence_header: String,
}

impl Default for Prompt {
    fn default() -> Self {
        Self {
            system: None,
            system_file: None,
            role_line: "The user is `{user}`. Roles: {roles}.".into(),
            role_line_no_roles: "The user is `{user}`. They hold no special roles.".into(),
            memory_header: "What you remember about this user:".into(),
            evidence_header: "Evidence from ASU sources. Answer only from this; cite sources by \
                              number. If it does not answer the question, say so."
                .into(),
        }
    }
}

/// What the risk policy allows, denies, and holds for confirmation.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct PolicyRules {
    /// Roles allowed to run `external_write` and above. Empty denies everyone.
    pub write_roles: Vec<String>,
    /// Let tools read inside the user's own authenticated session.
    pub allow_authenticated_reads: bool,
    /// Lowest risk class that must be confirmed before it runs.
    pub confirm_from: RiskClass,
}

impl Default for PolicyRules {
    fn default() -> Self {
        Self {
            write_roles: vec!["MANAGE_GUILD".into()],
            allow_authenticated_reads: false,
            confirm_from: RiskClass::ExternalWrite,
        }
    }
}

/// Which tools are registered. A tool that is not registered is one the model never sees.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Tools {
    /// Tool names never registered, whatever their source.
    pub disabled: Vec<String>,
    /// Register the built-in retrieval tool.
    pub knowledge_search: bool,
    /// Register the live source-query tool, when the scraper has published a registry.
    pub query_source: bool,
    /// Budget for one live query, end to end. A live fetch is far slower than a database read,
    /// so this overrides `agent.tool_timeout_secs` for this tool.
    pub query_timeout_secs: u64,
    /// How often the engine checks whether the worker has answered.
    pub query_poll_ms: u64,
}

impl Default for Tools {
    fn default() -> Self {
        Self {
            disabled: Vec::new(),
            knowledge_search: true,
            query_source: true,
            query_timeout_secs: 90,
            query_poll_ms: 400,
        }
    }
}

/// Hybrid retrieval tuning. Dense and lexical are fused with reciprocal rank fusion.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Retrieval {
    /// Evidence chunks handed to the prompt per request.
    pub top_k: usize,
    /// Candidates pulled from each leg before fusion. Raise it for recall, at query cost.
    pub candidates: i64,
    /// Reciprocal rank fusion constant. Lower trusts the top of each list more.
    pub rrf_k: f32,
    /// `PostgreSQL` text search configuration for the lexical leg.
    pub text_search_config: String,
    /// Run the pgvector leg.
    pub dense: bool,
    /// Run the full-text leg.
    pub lexical: bool,
    /// Drop fused results below this score. Zero keeps everything.
    pub min_score: f32,
}

impl Default for Retrieval {
    fn default() -> Self {
        Self {
            top_k: 6,
            candidates: 20,
            rrf_k: 60.0,
            text_search_config: "english".into(),
            dense: true,
            lexical: true,
            min_score: 0.0,
        }
    }
}

/// JSONL trace recording. One file per request under `dir`.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Trace {
    /// Write traces at all.
    pub enabled: bool,
    /// Directory for JSONL traces.
    pub dir: String,
    /// Stop writing a request's trace past this many bytes. 0 removes the limit.
    pub max_file_bytes: u64,
    /// Delete traces older than this at boot. 0 keeps them forever.
    pub retention_hours: u64,
}

impl Default for Trace {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: ".sparky/traces".into(),
            max_file_bytes: 0,
            retention_hours: 0,
        }
    }
}

/// HTTP surface limits.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Http {
    /// Largest request body accepted, in bytes.
    pub max_body_bytes: usize,
    /// Requests handled at once. 0 removes the limit.
    pub concurrency_limit: usize,
    /// Requests one user may start per minute. 0 removes the limit.
    pub rate_limit_per_min: u32,
    /// Origins allowed to call the engine from a browser. A single `*` allows any; empty adds
    /// no CORS headers, which is right when only the bot calls the engine.
    pub cors_origins: Vec<String>,
    /// How long in-flight requests get to finish after a shutdown signal.
    pub shutdown_grace_secs: u64,
}

impl Default for Http {
    fn default() -> Self {
        Self {
            max_body_bytes: 1 << 20,
            concurrency_limit: 0,
            rate_limit_per_min: 0,
            cors_origins: Vec::new(),
            shutdown_grace_secs: 10,
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
    /// How long a caller waits for a pooled connection.
    #[serde(default = "default_acquire_timeout_secs")]
    pub acquire_timeout_secs: u64,
}

fn default_max_connections() -> u32 {
    8
}

fn default_acquire_timeout_secs() -> u64 {
    5
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
    /// `service.name` on exported spans. Defaults to the binary's own name.
    pub service_name: Option<String>,
    /// Fraction of traces exported, 0.0 to 1.0.
    pub sample_ratio: f64,
    /// Budget for one export batch.
    pub export_timeout_secs: u64,
    /// Only spans whose target starts with this are exported. Defaults to the binary's name,
    /// which keeps dependency spans out of Phoenix.
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

impl Config {
    /// Loads the TOML layer then `SPARKY_*` variables, `__` separating nesting:
    /// `SPARKY_POSTGRES__URL`. Environment values win over the file.
    pub fn load() -> Result<Self, ConfigError> {
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
        if self.agent.chars_per_token == 0 {
            return invalid("agent.chars_per_token must be at least 1".into());
        }
        if self.agent.prompt_budget_tokens == 0 {
            return invalid("agent.prompt_budget_tokens must be at least 1".into());
        }
        if self.agent.max_steps == 0 {
            return invalid("agent.max_steps must be at least 1".into());
        }
        if !(0.0..=1.0).contains(&self.telemetry.sample_ratio) {
            return invalid(format!(
                "telemetry.sample_ratio must be between 0 and 1, got {}",
                self.telemetry.sample_ratio
            ));
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
        if self.tools.query_source && self.tools.query_poll_ms == 0 {
            return invalid("tools.query_poll_ms must be at least 1".into());
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
        // The prompt itself is never trimmed, so a section budget above the total is a
        // configuration error rather than something assembly quietly ignores.
        for (name, value) in [
            ("evidence", self.agent.evidence_budget_tokens),
            ("history", self.agent.history_budget_tokens),
            ("memory", self.agent.memory_budget_tokens),
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

    /// System instructions: `prompt.system_file`, else `prompt.system`, else `fallback`.
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
