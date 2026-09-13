//! Chat and embeddings via Rig's OpenAI client at llama-server. Maps Rig types to core types.

use ::rig_core::client::BearerAuth;
use ::rig_core::completion::{
    AssistantContent, CompletionError, CompletionModel as _, CompletionRequest,
    FinishReason as RigFinish, ToolDefinition as RigTool,
};
use ::rig_core::embeddings::EmbeddingModel as _;
use ::rig_core::message::{Message as RigMessage, ReasoningContent, ToolChoice, UserContent};
use ::rig_core::providers::openai::{CompletionModel, CompletionsClient, GenericEmbeddingModel};
use ::rig_core::streaming::StreamedAssistantContent;
use async_trait::async_trait;
use futures::StreamExt as _;
use secrecy::{ExposeSecret, SecretString};
use tokio::sync::mpsc::UnboundedSender;

use crate::core::config::{CHAT_TEMPLATE_KWARGS, ENABLE_THINKING};
use crate::core::traits::knowledge::retrieval::Embedder;
use crate::core::traits::model::ModelProvider;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::message::{Message, Role, ToolCall};
use crate::core::types::knowledge::retrieval::RetrievalError;
use crate::core::types::model::{
    FinishReason, ModelDelta, ModelError, ModelRequest, ModelResponse, Usage,
};
use crate::core::types::tools::ToolDefinition;

/// Builds a Rig client for one OpenAI-compatible base URL (ending in /v1).
pub fn client(base_url: &str, api_key: &SecretString) -> Result<CompletionsClient, String> {
    CompletionsClient::builder()
        .api_key(BearerAuth::from(api_key.expose_secret().to_owned()))
        .base_url(base_url.trim_end_matches('/'))
        .build()
        .map_err(|e| e.to_string())
}

/// ModelProvider over the chat-completions model of Rig.
#[derive(Clone)]
pub struct RigChat {
    model: CompletionModel,
    name: String,
    /// Provider-specific request fields sent with every completion.
    additional_params: serde_json::Value,
}

impl RigChat {
    /// Wraps model_name on client. additional_params is sent verbatim with every completion.
    pub fn new(
        client: CompletionsClient,
        model_name: impl Into<String>,
        additional_params: serde_json::Value,
    ) -> Self {
        let name = model_name.into();
        Self {
            model: CompletionModel::new(client, name.clone()),
            name,
            additional_params,
        }
    }
}

/// The provider request fields for one call: params with the chat template switch set to thinking.
pub(crate) fn with_thinking(params: &serde_json::Value, thinking: bool) -> serde_json::Value {
    let mut params = params.clone();
    if let serde_json::Value::Object(map) = &mut params {
        let kwargs = map
            .entry(CHAT_TEMPLATE_KWARGS)
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        if let serde_json::Value::Object(kwargs) = kwargs {
            kwargs.insert(
                ENABLE_THINKING.to_owned(),
                serde_json::Value::Bool(thinking),
            );
        }
    }
    params
}

/// Splits the flat message list into the Rig preamble plus chat history.
pub(crate) fn to_rig(
    messages: &[Message],
) -> Result<(Option<String>, Vec<RigMessage>), ModelError> {
    let mut preamble: Vec<String> = Vec::new();
    let mut history = Vec::with_capacity(messages.len());
    for m in messages {
        match m.role {
            Role::System => preamble.push(m.content.clone()),
            // A summary joins the preamble as operator context.
            Role::Summary => preamble.push(format!("Summary of earlier turns: {}", m.content)),
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

/// Maps Rig response content into core types: text, reasoning, and calls. Media is dropped.
pub(crate) fn from_rig(choice: Vec<AssistantContent>) -> (String, String, Vec<ToolCall>) {
    let mut text = String::new();
    let mut reasoning = String::new();
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
            AssistantContent::Reasoning(r) => {
                for block in r.content {
                    if let ReasoningContent::Text { text, .. } | ReasoningContent::Summary(text) =
                        block
                    {
                        if !reasoning.is_empty() {
                            reasoning.push('\n');
                        }
                        reasoning.push_str(&text);
                    }
                }
            }
            AssistantContent::Image(_) => {}
        }
    }
    (text, reasoning, calls)
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

impl RigChat {
    /// The Rig request for req.
    fn request(&self, req: &ModelRequest) -> Result<CompletionRequest, ModelError> {
        let (preamble, chat_history) = to_rig(&req.messages)?;
        let tools: Vec<RigTool> = req.tools.iter().map(tool_to_rig).collect();
        Ok(CompletionRequest {
            model: None,
            preamble,
            chat_history,
            documents: Vec::new(),
            tool_choice: (!tools.is_empty()).then_some(ToolChoice::Auto),
            tools,
            temperature: Some(f64::from(req.temperature)),
            max_tokens: Some(u64::from(req.max_tokens)),
            // llama-server honours chat_template_kwargs and the llama.cpp sampling fields.
            additional_params: Some(with_thinking(&self.additional_params, req.thinking)),
            output_schema: None,
            record_telemetry_content: false,
        })
    }

    /// The core response for a Rig choice, its usage, and the finish reason the provider gave.
    fn response(
        &self,
        choice: Vec<AssistantContent>,
        usage: ::rig_core::completion::Usage,
        reported: Option<&RigFinish>,
    ) -> ModelResponse {
        let (content, reasoning, tool_calls) = from_rig(choice);
        let finish_reason = finish_reason(reported, &content, &tool_calls);
        ModelResponse {
            content,
            reasoning,
            tool_calls,
            finish_reason,
            usage: Usage {
                prompt_tokens: u32::try_from(usage.input_tokens).unwrap_or(u32::MAX),
                completion_tokens: u32::try_from(usage.output_tokens).unwrap_or(u32::MAX),
            },
            model: self.name.clone(),
        }
    }
}

/// Why a completion stopped: what the provider reported, or else what the response shows.
pub(crate) fn finish_reason(
    reported: Option<&RigFinish>,
    content: &str,
    tool_calls: &[ToolCall],
) -> FinishReason {
    match reported {
        _ if !tool_calls.is_empty() => FinishReason::ToolCalls,
        Some(RigFinish::Length) => FinishReason::Length,
        Some(RigFinish::ToolCalls) => FinishReason::ToolCalls,
        Some(RigFinish::ContentFilter | RigFinish::Other(_)) => FinishReason::Other,
        None if content.trim().is_empty() => FinishReason::Unknown,
        Some(RigFinish::Stop) | None => FinishReason::Stop,
    }
}

#[async_trait]
impl ModelProvider for RigChat {
    async fn generate(
        &self,
        _ctx: &RequestContext,
        req: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        let request = self.request(&req)?;
        let response = self.model.completion(request).await.map_err(map_error)?;
        Ok(self.response(response.choice, response.usage, None))
    }

    async fn stream(
        &self,
        _ctx: &RequestContext,
        req: ModelRequest,
        deltas: UnboundedSender<ModelDelta>,
    ) -> Result<ModelResponse, ModelError> {
        let request = self.request(&req)?;
        let mut stream = self.model.stream(request).await.map_err(map_error)?;
        let mut usage = ::rig_core::completion::Usage::new();
        let mut reported = None;
        let mut finished = false;
        while let Some(item) = stream.next().await {
            // A closed receiver means nobody is watching; the completion continues.
            match item.map_err(map_error)? {
                StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                    let _ = deltas.send(ModelDelta::Reasoning(reasoning));
                }
                StreamedAssistantContent::Text(text) => {
                    let _ = deltas.send(ModelDelta::Text(text.text));
                }
                StreamedAssistantContent::Final(last) => {
                    usage = last.usage;
                    reported = last.finish_reason;
                    finished = true;
                }
                _ => {}
            }
        }
        if !finished {
            return Err(ModelError::Transport(
                "the model stream ended before the completion finished".into(),
            ));
        }
        let choice = std::mem::take(&mut stream.choice);
        Ok(self.response(choice, usage, reported.as_ref()))
    }
}

/// Embedder over the OpenAI-compatible embeddings model of Rig.
#[derive(Clone)]
pub struct RigEmbedder {
    model: GenericEmbeddingModel<::rig_core::providers::openai::OpenAICompletionsExt>,
    dim: usize,
}

impl RigEmbedder {
    /// Wraps model_name on client. dim must match the index the scraper wrote.
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
            // The index stores f32.
            #[allow(clippy::cast_possible_truncation)]
            out.push(e.vec.into_iter().map(|x| x as f32).collect());
        }
        Ok(out)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}
