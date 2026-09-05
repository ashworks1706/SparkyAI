//! `POST /v1/chat/completions` and `GET /v1/models`: the agent behind an `OpenAI`-compatible API,
//! so any off-the-shelf chat client drives the full loop instead of talking to the raw model.

use std::fmt::Write;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::core::types::agent::{AgentError, Answer};
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::model::ModelError;
use crate::core::types::openai::{
    ChatMessage, Choice, CompletionRequest, CompletionResponse, CompletionUsage, ModelCard,
    ModelList,
};
use crate::core::types::trace::RunStatus;
use crate::routes::chat::{ChatState, authorized};

/// The name the engine answers as. The agent may call several models in one run, so this names
/// the agent rather than any one of them.
pub const MODEL: &str = "sparky";

/// Namespace for conversation ids derived from a client's first message.
const CONVERSATION_NS: Uuid = Uuid::from_u128(0x5041_524b_5941_4931_4f50_454e_4149_0001);

/// The newest user turn, the only one the engine needs: it keeps its own history.
pub fn last_user_message(messages: &[ChatMessage]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.trim())
        .filter(|text| !text.is_empty())
}

/// A conversation id that stays put while a chat grows. `OpenAI` clients resend the whole
/// history every turn and carry no conversation id, so it is derived from the caller and the
/// turn that opened the chat.
pub fn conversation_for(user: Option<&str>, first_message: &str) -> Uuid {
    let seed = format!("{}\u{0}{first_message}", user.unwrap_or_default());
    Uuid::new_v5(&CONVERSATION_NS, seed.as_bytes())
}

/// The answer as one block of text: `OpenAI`'s schema has nowhere to put tool runs or
/// citations, so they follow the answer the way the Discord client renders them.
pub fn transcript(answer: &Answer) -> String {
    let mut out = answer.text.trim().to_owned();
    if let Some(c) = &answer.confirmation {
        let _ = write!(
            out,
            "\n\nNot done — `{}` needs approval: {}",
            c.tool, c.summary
        );
    }
    if !answer.tool_runs.is_empty() {
        let ran: Vec<String> = answer
            .tool_runs
            .iter()
            .map(|t| {
                if t.ok {
                    t.tool.clone()
                } else {
                    format!("{} (failed)", t.tool)
                }
            })
            .collect();
        let _ = write!(out, "\n\nTools: {}", ran.join(" \u{2192} "));
    }
    if !answer.evidence.is_empty() {
        out.push_str("\n\nSources");
        for (i, e) in answer.evidence.iter().enumerate() {
            let _ = write!(out, "\n{}. {}", i + 1, Evidence::citation(e));
        }
    }
    out
}

fn finish_reason(status: &RunStatus) -> &'static str {
    match status {
        RunStatus::StepLimit => "length",
        _ => "stop",
    }
}

/// Lists the one model the engine answers as, so clients that probe first do not fail.
pub async fn models() -> Response {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: MODEL.to_owned(),
            object: "model",
            created: chrono::Utc::now().timestamp(),
            owned_by: "sparky",
        }],
    })
    .into_response()
}

/// Runs one turn for an OpenAI-compatible client.
pub async fn completions(
    State(state): State<ChatState>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    let Some(input) = last_user_message(&req.messages) else {
        return (StatusCode::BAD_REQUEST, "no user message").into_response();
    };
    let input = input.to_owned();
    let opener = req
        .messages
        .iter()
        .find(|m| m.role == "user")
        .map_or(input.as_str(), |m| m.content.trim())
        .to_owned();

    let user = req
        .user
        .clone()
        .unwrap_or_else(|| "openai-client".to_owned());
    let ctx = RequestContext::new(state.default_tenant.clone(), user, state.request_budget)
        .with_conversation(conversation_for(req.user.as_deref(), &opener));

    if let Some(store) = &state.conversations
        && let Err(e) = store.ensure(&ctx, "openai").await
    {
        tracing::error!(error = %e, "conversation store unavailable");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "conversation store unavailable",
        )
            .into_response();
    }

    let answer = match state.agent.run(&ctx, &input).await {
        Ok(answer) => answer,
        Err(AgentError::Model(ModelError::Busy)) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "the model is at capacity").into_response();
        }
        Err(AgentError::Model(e)) => {
            tracing::error!(error = %e, "model failed");
            return (StatusCode::BAD_GATEWAY, "the model is unavailable").into_response();
        }
        Err(AgentError::Store(e)) => {
            tracing::error!(error = %e, "store failed");
            return (StatusCode::SERVICE_UNAVAILABLE, "a store is unavailable").into_response();
        }
    };

    let content = transcript(&answer);
    let reason = finish_reason(&answer.status);
    let id = format!("chatcmpl-{}", ctx.request_id);
    let created = chrono::Utc::now().timestamp();

    if req.stream {
        // The engine does not stream tokens, so the answer arrives as a single delta. Clients
        // that require SSE work; nobody is told a token arrived before it did.
        let first = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                         "finish_reason": Value::Null}],
        });
        let last = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
        });
        let events = vec![
            Ok::<Event, std::convert::Infallible>(Event::default().data(first.to_string())),
            Ok(Event::default().data(last.to_string())),
            Ok(Event::default().data("[DONE]")),
        ];
        return Sse::new(futures::stream::iter(events)).into_response();
    }

    Json(CompletionResponse {
        id,
        object: "chat.completion",
        created,
        model: MODEL.to_owned(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason: reason.to_owned(),
        }],
        usage: CompletionUsage {
            prompt_tokens: answer.usage.prompt_tokens,
            completion_tokens: answer.usage.completion_tokens,
            total_tokens: answer.usage.total(),
        },
    })
    .into_response()
}
