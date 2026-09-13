//! POST /v1/chat/completions and GET /v1/models: agent behind an OpenAI API for any chat client.

use std::fmt::Write;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::agent::{AgentError, Answer};
use crate::core::types::conversation::Visibility;
use crate::core::types::http::openai::{
    ChatMessage, Choice, CompletionRequest, CompletionResponse, CompletionUsage, ModelCard,
    ModelList,
};
use crate::core::types::model::ModelError;
use crate::core::types::store::StoreError;
use crate::core::types::trace::RunStatus;
use crate::routes::chat::{ChatState, NO_SUCH_CONVERSATION, authorized, too_many};

/// The name the engine answers as. It names the agent, not any one model.
pub const MODEL: &str = "sparky";

/// Namespace for conversation ids derived from the first message of a client.
const CONVERSATION_NS: Uuid = Uuid::from_u128(0x5041_524b_5941_4931_4f50_454e_4149_0001);

/// The newest user turn. The engine keeps its own history.
pub fn last_user_message(messages: &[ChatMessage]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.trim())
        .filter(|text| !text.is_empty())
}

/// A conversation id that stays put as a chat grows, derived from the caller and the opening turn.
pub fn conversation_for(user: &str, first_message: &str) -> Uuid {
    let seed = format!("{user}\u{0}{first_message}");
    Uuid::new_v5(&CONVERSATION_NS, seed.as_bytes())
}

/// The answer as one block of text, with tool runs and citations following it.
pub fn transcript(answer: &Answer) -> String {
    let mut out = answer.text.trim().to_owned();
    // Writing to a String cannot fail.
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
    let citations = answer.citations();
    if !citations.is_empty() {
        out.push_str("\n\nSources");
        for (i, source) in citations.iter().enumerate() {
            let _ = write!(out, "\n{}. {}", i + 1, source.line());
        }
    }
    out
}

/// The OpenAI finish_reason for how a run ended.
fn finish_reason(status: &RunStatus) -> &'static str {
    match status {
        RunStatus::StepLimit => "length",
        _ => "stop",
    }
}

/// A streamed completion holding the whole answer as one delta, then the finish reason.
fn single_delta(id: &str, created: i64, content: &str, reason: &str) -> Response {
    let first = json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": MODEL,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                     "finish_reason": Value::Null}],
    });
    let last = json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    });
    let events = [
        Ok::<Event, std::convert::Infallible>(Event::default().data(first.to_string())),
        Ok(Event::default().data(last.to_string())),
        Ok(Event::default().data("[DONE]")),
    ];
    Sse::new(futures::stream::iter(events)).into_response()
}

/// Lists the one model the engine answers as.
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

    // Conversation and memory are keyed by the caller named in user.
    let Some(user) = req.user.as_deref().map(str::trim).filter(|u| !u.is_empty()) else {
        return (
            StatusCode::BAD_REQUEST,
            "set `user` on the request: the engine keeps conversation and memory per caller",
        )
            .into_response();
    };
    if !state.rate_limit.allow(user) {
        return too_many(user);
    }
    let ctx = RequestContext::new(state.default_tenant.clone(), user, state.request_budget)
        .with_conversation(conversation_for(user, &opener))
        .with_visibility(Visibility::Private);

    if let Some(store) = &state.conversations {
        match store.ensure(&ctx, "openai").await {
            Ok(()) => {}
            Err(StoreError::NotOwned) => {
                return (StatusCode::NOT_FOUND, NO_SUCH_CONVERSATION).into_response();
            }
            Err(e) => {
                tracing::error!(error = %e, "conversation store unavailable");
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "conversation store unavailable",
                )
                    .into_response();
            }
        }
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
        return single_delta(&id, created, &content, reason);
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
