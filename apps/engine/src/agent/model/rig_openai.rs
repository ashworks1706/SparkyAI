//! Chat and embeddings over Rig's OpenAI-compatible client, pointed at llama-server.
//! Rig owns the wire format; this module maps between its types and `core::types`.

use ::rig_core::client::BearerAuth;
use ::rig_core::completion::{
    AssistantContent, CompletionError, CompletionModel as _, CompletionRequest,
    ToolDefinition as RigTool,
};
use ::rig_core::embeddings::EmbeddingModel as _;
use ::rig_core::message::{Message as RigMessage, ToolChoice, UserContent};
use ::rig_core::providers::openai::{CompletionModel, CompletionsClient, GenericEmbeddingModel};
use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};

use crate::core::traits::model::ModelProvider;
use crate::core::traits::retrieval::Embedder;
use crate::core::types::context::RequestContext;
use crate::core::types::message::{Message, Role, ToolCall};
use crate::core::types::model::{FinishReason, ModelError, ModelRequest, ModelResponse, Usage};
use crate::core::types::retrieval::RetrievalError;
use crate::core::types::tool::ToolDefinition;

/// Builds a Rig client for one OpenAI-compatible base URL (ending in `/v1`).
pub fn client(base_url: &str, api_key: &SecretString) -> Result<CompletionsClient, String> {
    CompletionsClient::builder()
        .api_key(BearerAuth::from(api_key.expose_secret().to_owned()))
        .base_url(base_url.trim_end_matches('/'))
        .build()
        .map_err(|e| e.to_string())
}

/// `ModelProvider` over Rig's chat-completions model.
#[derive(Clone)]
pub struct RigChat {
    model: CompletionModel,
    name: String,
    thinking: bool,
}

impl RigChat {
    /// Wraps `model_name` on `client`.
    pub fn new(client: CompletionsClient, model_name: impl Into<String>, thinking: bool) -> Self {
        let name = model_name.into();
        Self {
            model: CompletionModel::new(client, name.clone()),
            name,
            thinking,
        }
    }
}

/// Splits our flat message list into Rig's preamble plus chat history.
/// Fails on a tool result that carries no call id or tool name: assembly produced a history the
/// provider cannot represent.
pub(crate) fn to_rig(
    messages: &[Message],
) -> Result<(Option<String>, Vec<RigMessage>), ModelError> {
    let mut preamble: Vec<&str> = Vec::new();
    let mut history = Vec::with_capacity(messages.len());
    for m in messages {
        match m.role {
            Role::System => preamble.push(&m.content),
            Role::User => history.push(RigMessage::user(&m.content)),
            Role::Assistant => {
                let mut content = Vec::with_capacity(m.tool_calls.len() + 1);
                if !m.content.is_empty() {
                    content.push(AssistantContent::text(&m.content));
                }
                for call in &m.tool_calls {
                    content.push(AssistantContent::tool_call(
                        &call.id,
                        &call.name,
                        call.arguments.clone(),
                    ));
                }
                history.push(RigMessage::Assistant { id: None, content });
            }
            Role::Tool => {
                let (Some(call_id), Some(name)) =
                    (m.tool_call_id.as_deref(), m.tool_name.as_deref())
                else {
                    return Err(ModelError::Malformed(
                        "tool result without call id or tool name".into(),
                    ));
                };
                history.push(RigMessage::User {
                    content: vec![UserContent::tool_result_from_wire(
                        call_id,
                        name,
                        vec![::rig_core::message::ToolResultContent::text(&m.content)],
                    )],
                });
            }
        }
    }
    let preamble = (!preamble.is_empty()).then(|| preamble.join("\n\n"));
    Ok((preamble, history))
}

fn tool_to_rig(t: &ToolDefinition) -> RigTool {
    RigTool {
        name: t.name.clone(),
        description: t.description.clone(),
        parameters: t.parameters.clone(),
    }
}

/// Maps Rig's response content into ours. Reasoning and media are dropped.
pub(crate) fn from_rig(choice: Vec<AssistantContent>) -> (String, Vec<ToolCall>) {
    let mut text = String::new();
    let mut calls = Vec::new();
    for item in choice {
        match item {
            AssistantContent::Text(t) => text.push_str(&t.text),
            AssistantContent::ToolCall(call) => {
                let id = call
                    .provider
                    .as_ref()
                    .map_or_else(|| call.id.to_string(), |p| p.call_id.clone());
                calls.push(ToolCall {
                    id,
                    name: call.function.name,
                    arguments: call.function.arguments,
                });
            }
            AssistantContent::Reasoning(_) | AssistantContent::Image(_) => {}
        }
    }
    (text, calls)
}

fn map_error(e: CompletionError) -> ModelError {
    use ::rig_core::http_client::Error as Http;
    match e {
        CompletionError::HttpError(http) => match http {
            Http::InvalidStatusCode(status) => ModelError::Status {
                status: status.as_u16(),
                body: String::new(),
            },
            Http::InvalidStatusCodeWithMessage(status, body) => ModelError::Status {
                status: status.as_u16(),
                body: body.chars().take(500).collect(),
            },
            other => {
                let text = other.to_string();
                if text.contains("timed out") {
                    ModelError::Timeout
                } else {
                    ModelError::Transport(text)
                }
            }
        },
        CompletionError::JsonError(err) => ModelError::Malformed(err.to_string()),
        CompletionError::ResponseError(text) | CompletionError::ProviderError(text) => {
            ModelError::Transport(text)
        }
        other => ModelError::Malformed(other.to_string()),
    }
}

#[async_trait]
impl ModelProvider for RigChat {
    async fn generate(
        &self,
        _ctx: &RequestContext,
        req: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        let (preamble, chat_history) = to_rig(&req.messages)?;
        let tools: Vec<RigTool> = req.tools.iter().map(tool_to_rig).collect();
        let request = CompletionRequest {
            model: None,
            preamble,
            chat_history,
            documents: Vec::new(),
            tool_choice: (!tools.is_empty()).then_some(ToolChoice::Auto),
            tools,
            temperature: Some(f64::from(req.temperature)),
            max_tokens: Some(u64::from(req.max_tokens)),
            // llama-server honours chat_template_kwargs; this switches Qwen3 thinking off.
            additional_params: Some(serde_json::json!({
                "chat_template_kwargs": { "enable_thinking": self.thinking }
            })),
            output_schema: None,
            record_telemetry_content: false,
        };
        let response = self.model.completion(request).await.map_err(map_error)?;
        let (content, tool_calls) = from_rig(response.choice);
        // Rig does not surface the provider's finish reason. Tool calls and text are the two
        // cases it can be read off the response; an empty completion is not guessed at, since
        // it can be a length cut, a refusal, or reasoning the adapter dropped.
        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else if content.trim().is_empty() {
            FinishReason::Unknown
        } else {
            FinishReason::Stop
        };
        Ok(ModelResponse {
            content,
            tool_calls,
            finish_reason,
            usage: Usage {
                prompt_tokens: u32::try_from(response.usage.input_tokens).unwrap_or(u32::MAX),
                completion_tokens: u32::try_from(response.usage.output_tokens).unwrap_or(u32::MAX),
            },
            model: self.name.clone(),
        })
    }
}

/// `Embedder` over Rig's OpenAI-compatible embeddings model.
#[derive(Clone)]
pub struct RigEmbedder {
    model: GenericEmbeddingModel<::rig_core::providers::openai::OpenAICompletionsExt>,
    dim: usize,
}

impl RigEmbedder {
    /// Wraps `model_name` on `client`; `dim` must match the index the scraper wrote.
    pub fn new(client: CompletionsClient, model_name: impl Into<String>, dim: usize) -> Self {
        Self {
            model: GenericEmbeddingModel::new(client, model_name, dim),
            dim,
        }
    }
}

#[async_trait]
impl Embedder for RigEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, RetrievalError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let embeddings = self
            .model
            .embed_texts(texts.iter().cloned())
            .await
            .map_err(|e| RetrievalError::Embedding(e.to_string()))?;
        if embeddings.len() != texts.len() {
            return Err(RetrievalError::Embedding(format!(
                "asked for {} vectors, got {}",
                texts.len(),
                embeddings.len()
            )));
        }
        let mut out = Vec::with_capacity(embeddings.len());
        for e in embeddings {
            if e.vec.len() != self.dim {
                return Err(RetrievalError::Embedding(format!(
                    "dimension {} does not match configured {}",
                    e.vec.len(),
                    self.dim
                )));
            }
            // f64 → f32 narrowing is intended; the index stores f32.
            #[allow(clippy::cast_possible_truncation)]
            out.push(e.vec.into_iter().map(|x| x as f32).collect());
        }
        Ok(out)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
