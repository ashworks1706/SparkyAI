//! OpenAI-compatible chat completions client (llama-server). Tool calls and usage.

use std::time::Duration;

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::harness::model::{
    FinishReason, ModelError, ModelProvider, ModelRequest, ModelResponse, Usage,
};
use crate::agent::harness::tool::ToolDefinition;
use crate::core::types::context::RequestContext;
use crate::core::types::message::{Message, Role, ToolCall};

/// Chat client for one model behind `/v1/chat/completions`.
#[derive(Debug, Clone)]
pub struct OpenAiCompat {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
    model: String,
}

impl OpenAiCompat {
    /// Builds a client. `base_url` ends in `/v1`.
    pub fn new(
        base_url: impl Into<String>,
        api_key: SecretString,
        model: impl Into<String>,
    ) -> Result<Self, ModelError> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
            model: model.into(),
        })
    }
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: Role,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCallOut<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

#[derive(Serialize)]
struct WireToolCallOut<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireFunctionOut<'a>,
}

#[derive(Serialize)]
struct WireFunctionOut<'a> {
    name: &'a str,
    arguments: String,
}

#[derive(Serialize)]
struct WireTool<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireToolFunction<'a>,
}

#[derive(Serialize)]
struct WireToolFunction<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a Value,
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
}

#[derive(Deserialize)]
struct WireResponse {
    #[serde(default)]
    model: String,
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireChoice {
    message: WireMessageIn,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireMessageIn {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCallIn>,
}

#[derive(Deserialize)]
struct WireToolCallIn {
    #[serde(default)]
    id: Option<String>,
    function: WireFunctionIn,
}

#[derive(Deserialize)]
struct WireFunctionIn {
    name: String,
    #[serde(default)]
    arguments: Option<Value>,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

fn to_wire(m: &Message) -> WireMessage<'_> {
    WireMessage {
        role: m.role,
        content: &m.content,
        tool_calls: (!m.tool_calls.is_empty()).then(|| {
            m.tool_calls
                .iter()
                .map(|c| WireToolCallOut {
                    id: &c.id,
                    kind: "function",
                    function: WireFunctionOut {
                        name: &c.name,
                        arguments: c.arguments.to_string(),
                    },
                })
                .collect()
        }),
        tool_call_id: m.tool_call_id.as_deref(),
    }
}

fn tool_to_wire(t: &ToolDefinition) -> WireTool<'_> {
    WireTool {
        kind: "function",
        function: WireToolFunction {
            name: &t.name,
            description: &t.description,
            parameters: &t.parameters,
        },
    }
}

/// Arguments arrive either as a JSON string or as an object, depending on the server.
fn parse_arguments(v: Option<Value>) -> Result<Value, ModelError> {
    match v {
        None => Ok(Value::Object(serde_json::Map::new())),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(Value::Object(serde_json::Map::new())),
        Some(Value::String(s)) => serde_json::from_str(&s)
            .map_err(|e| ModelError::Malformed(format!("tool arguments: {e}"))),
        Some(other) => Ok(other),
    }
}

fn finish(reason: Option<&str>, has_tool_calls: bool) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        _ if has_tool_calls => FinishReason::ToolCalls,
        None => FinishReason::Stop,
        Some(_) => FinishReason::Other,
    }
}

#[async_trait]
impl ModelProvider for OpenAiCompat {
    async fn generate(
        &self,
        ctx: &RequestContext,
        req: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        let body = WireRequest {
            model: &self.model,
            messages: req.messages.iter().map(to_wire).collect(),
            tools: req.tools.iter().map(tool_to_wire).collect(),
            tool_choice: (!req.tools.is_empty()).then_some("auto"),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: false,
        };
        let mut request = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .timeout(ctx.remaining())
            .json(&body);
        if !self.api_key.expose_secret().is_empty() {
            request = request.bearer_auth(self.api_key.expose_secret());
        }
        let response = request.send().await.map_err(|e| {
            if e.is_timeout() {
                ModelError::Timeout
            } else {
                ModelError::Transport(e.to_string())
            }
        })?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        if !status.is_success() {
            return Err(ModelError::Status {
                status: status.as_u16(),
                body: text.chars().take(500).collect(),
            });
        }
        let wire: WireResponse =
            serde_json::from_str(&text).map_err(|e| ModelError::Malformed(e.to_string()))?;
        let choice = wire
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ModelError::Malformed("no choices".into()))?;
        let tool_calls = choice
            .message
            .tool_calls
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                Ok(ToolCall {
                    id: c.id.unwrap_or_else(|| format!("call_{i}")),
                    name: c.function.name,
                    arguments: parse_arguments(c.function.arguments)?,
                })
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        let finish_reason = finish(choice.finish_reason.as_deref(), !tool_calls.is_empty());
        let usage = wire.usage.map_or_else(Usage::default, |u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
        });
        Ok(ModelResponse {
            content: choice.message.content.unwrap_or_default(),
            tool_calls,
            finish_reason,
            usage,
            model: if wire.model.is_empty() {
                self.model.clone()
            } else {
                wire.model
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn arguments_accept_string_or_object() {
        assert_eq!(
            parse_arguments(Some(json!("{\"a\":1}"))).ok(),
            Some(json!({"a": 1}))
        );
        assert_eq!(
            parse_arguments(Some(json!({"a": 1}))).ok(),
            Some(json!({"a": 1}))
        );
        assert_eq!(parse_arguments(None).ok(), Some(json!({})));
        assert!(parse_arguments(Some(json!("not json"))).is_err());
    }

    #[test]
    fn finish_reason_falls_back_on_tool_calls() {
        assert_eq!(finish(None, true), FinishReason::ToolCalls);
        assert_eq!(finish(Some("stop"), false), FinishReason::Stop);
        assert_eq!(finish(Some("length"), false), FinishReason::Length);
    }
}
