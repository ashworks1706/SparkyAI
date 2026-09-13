//! Settings that name an external endpoint or a credential.

use secrecy::SecretString;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::core::config::ConfigError;

/// Process-level settings.
#[derive(Debug, Deserialize)]
pub struct App {
    /// development, staging, or production.
    pub env: String,
    /// Bind address for the HTTP server.
    pub http_addr: String,
    /// tracing filter directive, e.g. info,sparky=debug.
    pub log_level: String,
}

/// Client authentication.
#[derive(Debug, Deserialize)]
pub struct Engine {
    /// Bearer token every /chat caller must present.
    pub service_token: SecretString,
}

/// The guild this engine serves.
#[derive(Debug, Deserialize)]
pub struct Discord {
    /// The one guild this deployment serves.
    pub guild_id: u64,
}

/// Provider sampling parameters. Only the fields that are set are sent.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Sampling {
    /// Nucleus sampling cutoff.
    pub top_p: Option<f64>,
    /// Keep only the k most likely tokens.
    pub top_k: Option<u32>,
    /// Drop tokens below this fraction of the probability of the top token.
    pub min_p: Option<f64>,
    /// Penalty applied to tokens already in the context.
    pub repeat_penalty: Option<f64>,
    /// OpenAI-style presence penalty.
    pub presence_penalty: Option<f64>,
    /// OpenAI-style frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// Fixes sampling. Set it for eval.
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
        for (key, value) in self.floats() {
            put(
                key,
                value
                    .and_then(serde_json::Number::from_f64)
                    .map(Value::Number),
            );
        }
        put("top_k", self.top_k.map(|v| Value::Number(v.into())));
        put("seed", self.seed.map(|v| Value::Number(v.into())));
        if !self.stop.is_empty() {
            map.insert("stop".to_owned(), Value::from(self.stop.clone()));
        }
        map
    }

    /// The floating point fields by request key.
    fn floats(&self) -> [(&'static str, Option<f64>); 5] {
        [
            ("top_p", self.top_p),
            ("min_p", self.min_p),
            ("repeat_penalty", self.repeat_penalty),
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ]
    }
}

/// Request field holding the chat template arguments.
pub const CHAT_TEMPLATE_KWARGS: &str = "chat_template_kwargs";
/// Chat template argument that turns reasoning on for one call.
pub const ENABLE_THINKING: &str = "enable_thinking";

/// Chat model served by llama-server (OpenAI-compatible).
#[derive(Debug, Deserialize)]
pub struct Model {
    /// OpenAI-compatible base URL, ending in /v1.
    pub base_url: String,
    /// API key for the endpoint.
    pub api_key: SecretString,
    /// Model name as served.
    pub name: String,
    /// Completion budget of a call that thinks: room for the reasoning and the answer.
    pub max_tokens: u32,
    /// Completion budget of a call that does not think.
    #[serde(default = "default_max_tokens_without_thinking")]
    pub max_tokens_without_thinking: u32,
    /// USD per million prompt tokens; zero for local serving.
    #[serde(default)]
    pub usd_per_m_prompt: f64,
    /// USD per million completion tokens; zero for local serving.
    #[serde(default)]
    pub usd_per_m_completion: f64,
    /// Sampling parameters sent with every completion.
    #[serde(default)]
    pub sampling: Sampling,
    /// Merged into the provider request over sampling as JSON. Thinking is set by agent.thinking.
    #[serde(default)]
    pub extra_params_json: Option<String>,
}

impl Model {
    /// Provider-specific request fields: sampling, then extra_params_json on top.
    pub fn additional_params(&self) -> Result<Value, ConfigError> {
        let mut map = self.sampling.to_json();
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
            match extra.get(CHAT_TEMPLATE_KWARGS) {
                None => {}
                Some(Value::Object(kwargs)) if !kwargs.contains_key(ENABLE_THINKING) => {}
                Some(Value::Object(_)) => {
                    return Err(ConfigError::Invalid(format!(
                        "model.extra_params_json sets {ENABLE_THINKING}; set agent.thinking.mode"
                    )));
                }
                Some(_) => {
                    return Err(ConfigError::Invalid(format!(
                        "model.extra_params_json {CHAT_TEMPLATE_KWARGS} must be a JSON object"
                    )));
                }
            }
            map.extend(extra);
        }
        Ok(Value::Object(map))
    }
}

/// PostgreSQL connection.
#[derive(Debug, Deserialize)]
pub struct Postgres {
    /// libpq connection URL.
    pub url: SecretString,
    /// Maximum pooled connections.
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    /// How long a caller waits for a pooled connection.
    #[serde(default = "default_acquire_timeout_secs")]
    pub acquire_timeout_secs: u64,
}

/// Rejects a model section leaving no room to answer, or with a non-finite sampling value.
pub fn validate_model(model: &Model) -> Result<(), ConfigError> {
    if model.max_tokens == 0 || model.max_tokens_without_thinking == 0 {
        return Err(ConfigError::Invalid(
            "model.max_tokens and model.max_tokens_without_thinking must be at least 1".into(),
        ));
    }
    if let Some((key, _)) = model
        .sampling
        .floats()
        .into_iter()
        .find(|(_, value)| value.is_some_and(|v| !v.is_finite()))
    {
        return Err(ConfigError::Invalid(format!(
            "model.sampling.{key} must be a finite number"
        )));
    }
    Ok(())
}

fn default_max_tokens_without_thinking() -> u32 {
    1_024
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
    /// Base URL, ending in /v1.
    pub base_url: String,
    /// API key for the endpoint.
    pub api_key: SecretString,
    /// Model name as served.
    pub name: String,
    /// Vector dimension; must match the chunks.embedding column.
    pub dim: u32,
}

/// Trace export to PostHog and Phoenix.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Telemetry {
    /// PostHog base URL; unset or empty disables trace export.
    pub host: Option<String>,
    /// PostHog project token; empty disables trace export.
    pub project_token: SecretString,
    /// Path of the OTLP traces endpoint, joined to host.
    pub traces_path: String,
    /// Path of the OTLP AI endpoint, joined to host. Empty exports nothing there.
    pub ai_path: String,
    /// Phoenix base URL for the trace UI; unset or empty exports nothing there.
    pub phoenix_url: Option<String>,
    /// gen_ai.provider.name on model spans.
    pub provider_name: String,
    /// service.name on exported spans. Defaults to the name of the binary.
    pub service_name: Option<String>,
    /// Fraction of traces exported, 0.0 to 1.0.
    pub sample_ratio: f64,
    /// Budget for one export batch.
    pub export_timeout_secs: u64,
    /// Only spans whose target starts with this are exported. Defaults to the name of the binary.
    pub span_target_prefix: Option<String>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            host: Some("http://localhost:8010".into()),
            project_token: SecretString::from(""),
            traces_path: "/i/v1/traces".into(),
            ai_path: String::new(),
            phoenix_url: None,
            provider_name: "llama.cpp".into(),
            service_name: None,
            sample_ratio: 1.0,
            export_timeout_secs: 10,
            span_target_prefix: None,
        }
    }
}
