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

/// Redis connection, shared by every engine replica that caches live queries.
#[derive(Debug, Deserialize)]
pub struct Redis {
    /// Connection URL, for example redis://localhost:6379.
    pub url: SecretString,
    /// Budget for connecting at boot.
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
}

fn default_connect_timeout_secs() -> u64 {
    5
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

/// Trace export to Phoenix.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Telemetry {
    /// Phoenix base URL; unset or empty disables trace export.
    pub phoenix_url: Option<String>,
    /// Bearer token for a Phoenix that authenticates. Empty sends no Authorization header.
    pub phoenix_api_key: SecretString,
    /// The Phoenix project exported spans land in.
    pub project_name: String,
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
            phoenix_url: None,
            phoenix_api_key: SecretString::from(""),
            project_name: "sparky".into(),
            provider_name: "llama.cpp".into(),
            service_name: None,
            sample_ratio: 1.0,
            export_timeout_secs: 10,
            span_target_prefix: None,
        }
    }
}

/// OAuth clients for per-user grants.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct OAuth {
    /// Google, for Calendar through its MCP server.
    pub google: GoogleOAuth,
}

/// Canvas LMS integration. Off by default. Read-only, in a direct message only.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Canvas {
    /// Whether the Canvas tools are registered at boot.
    pub enabled: bool,
    /// Base URL of the Canvas instance, no trailing path.
    pub base_url: String,
    /// Canvas API token used until per-user grants exist. Lives in .env.
    pub access_token: SecretString,
    /// Budget for one Canvas API request.
    pub timeout_secs: u64,
    /// Most rows one list tool returns to the model.
    pub max_items: usize,
}

impl Default for Canvas {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "https://canvas.asu.edu".into(),
            access_token: SecretString::from(""),
            timeout_secs: 30,
            max_items: 20,
        }
    }
}

/// A Google OAuth 2.0 web client. Off by default.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct GoogleOAuth {
    /// Whether the client is built at boot.
    pub enabled: bool,
    /// OAuth client id from the Google Cloud console.
    pub client_id: String,
    /// OAuth client secret. Lives in .env.
    pub client_secret: SecretString,
    /// Where Google sends the user back with a code.
    pub redirect_url: String,
    /// Scopes requested in one grant.
    pub scopes: Vec<String>,
    /// Google authorization endpoint.
    pub authorize_url: String,
    /// Google token endpoint.
    pub token_url: String,
    /// Budget for one token request.
    pub timeout_secs: u64,
}

impl Default for GoogleOAuth {
    fn default() -> Self {
        Self {
            enabled: false,
            client_id: String::new(),
            client_secret: SecretString::from(""),
            redirect_url: String::new(),
            scopes: vec!["https://www.googleapis.com/auth/calendar.events".into()],
            authorize_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token_url: "https://oauth2.googleapis.com/token".into(),
            timeout_secs: 30,
        }
    }
}
