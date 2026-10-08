//! Mapping between core types and Rig types: messages, tools, responses, errors, finish reasons.

use ::rig_core::completion::{
    AssistantContent, CompletionError, FinishReason as RigFinish, ToolDefinition as RigTool,
};
use ::rig_core::message::{
    DocumentSourceKind, Image as RigImage, ImageMediaType, Message as RigMessage, ReasoningContent,
    UserContent,
};

use crate::core::config::{CHAT_TEMPLATE_KWARGS, ENABLE_THINKING};
use crate::core::types::conversation::message::{Message, Role, ToolCall};
use crate::core::types::model::{FinishReason, ModelError};
use crate::core::types::tools::ToolDefinition;

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
            Role::User => history.push(user_message(m)),
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

/// A user turn, with an image block per attachment. Text only when it carries none.
fn user_message(m: &Message) -> RigMessage {
    if m.images.is_empty() {
        return RigMessage::user(&m.content);
    }
    let mut content = Vec::with_capacity(m.images.len() + 1);
    if !m.content.is_empty() {
        content.push(UserContent::Text(::rig_core::message::Text::new(
            &m.content,
        )));
    }
    content.extend(m.images.iter().map(|image| {
        UserContent::Image(RigImage {
            data: DocumentSourceKind::url(&image.url),
            media_type: media_type(&image.media_type),
            detail: None,
            additional_params: None,
        })
    }));
    match ::rig_core::message::non_empty(content) {
        Some(content) => RigMessage::User { content },
        None => RigMessage::user(&m.content),
    }
}

/// The Rig media type for one of IMAGE_MEDIA_TYPES. None lets the provider sniff it.
fn media_type(kind: &str) -> Option<ImageMediaType> {
    match kind {
        "image/png" => Some(ImageMediaType::PNG),
        "image/jpeg" => Some(ImageMediaType::JPEG),
        "image/gif" => Some(ImageMediaType::GIF),
        "image/webp" => Some(ImageMediaType::WEBP),
        _ => None,
    }
}

/// Maps a core tool definition to the Rig tool shape.
pub(super) fn tool_to_rig(t: &ToolDefinition) -> RigTool {
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

/// The ModelError a Rig completion error maps to.
pub(super) fn map_error(e: CompletionError) -> ModelError {
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
