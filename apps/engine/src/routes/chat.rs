//! `POST /chat`: one user message in, one answer with citations out.

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use secrecy::{ExposeSecret, SecretString};
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::agent::harness::agent::Agent;
use crate::core::traits::conversation::ConversationStore;
use crate::core::types::agent::AgentError;
use crate::core::types::chat::{ChatRequest, ChatResponse, ErrorBody};
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::model::ModelError;

/// What the chat route needs.
#[derive(Clone)]
pub struct ChatState {
    /// The agent.
    pub agent: Agent,
    /// To create the conversation row before the run.
    pub conversations: Option<Arc<dyn ConversationStore>>,
    /// Per-request wall-clock budget.
    pub request_budget: Duration,
    /// Tenant used when the client sends none (single-guild deployments).
    pub default_tenant: String,
    /// Bearer token every caller must present.
    pub service_token: SecretString,
}

fn error(status: StatusCode, ctx: &RequestContext, text: &str) -> Response {
    (
        status,
        Json(ErrorBody {
            request_id: ctx.request_id,
            error: text.into(),
        }),
    )
        .into_response()
}

/// Parses a W3C `traceparent` header into a remote parent context.
pub fn parse_traceparent(value: &str) -> Option<opentelemetry::Context> {
    let mut parts = value.trim().split('-');
    let (_version, trace_id, span_id, flags) =
        (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
    let trace_id = TraceId::from_hex(trace_id).ok()?;
    let span_id = SpanId::from_hex(span_id).ok()?;
    let flags = TraceFlags::new(u8::from_str_radix(flags, 16).ok()?);
    let remote = SpanContext::new(trace_id, span_id, flags, true, TraceState::default());
    remote
        .is_valid()
        .then(|| opentelemetry::Context::new().with_remote_span_context(remote))
}

/// Handles one chat turn. Runs under an `http.chat` span parented to the caller's
/// `traceparent`, so the bot's interaction and the engine's loop share one trace.
pub async fn chat(
    State(state): State<ChatState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    let span = tracing::info_span!("http.chat", "sparky.user_id" = %req.user_id);
    if let Some(parent) = headers
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_traceparent)
        && let Err(e) = span.set_parent(parent)
    {
        tracing::debug!(error = %e, "traceparent ignored");
    }
    chat_inner(state, req).instrument(span).await
}

async fn chat_inner(state: ChatState, req: ChatRequest) -> Response {
    if req.message.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "message is empty").into_response();
    }
    let tenant = req
        .tenant_id
        .unwrap_or_else(|| state.default_tenant.clone());
    let mut ctx =
        RequestContext::new(tenant, req.user_id, state.request_budget).with_roles(req.roles);
    if let Some(id) = req.conversation_id {
        ctx = ctx.with_conversation(id);
    }
    if let Some(store) = &state.conversations
        && let Err(e) = store.ensure(&ctx, &req.channel_id).await
    {
        tracing::error!(error = %e, "conversation store unavailable");
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            &ctx,
            "conversation store unavailable",
        );
    }
    match state.agent.run(&ctx, &req.message).await {
        Ok(answer) => Json(ChatResponse {
            request_id: ctx.request_id,
            conversation_id: ctx.conversation_id,
            citations: answer.evidence.iter().map(Evidence::citation).collect(),
            text: answer.text,
            confirmation: answer.confirmation,
            status: answer.status,
            steps: answer.steps,
            tools: answer.tool_runs,
            tokens: answer.usage.total(),
            cost_usd: answer.cost_usd,
        })
        .into_response(),
        Err(AgentError::Model(ModelError::Busy)) => {
            tracing::warn!(request_id = %ctx.request_id, "model at capacity");
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                &ctx,
                "the model is at capacity",
            )
        }
        Err(AgentError::Model(e)) => {
            tracing::error!(error = %e, request_id = %ctx.request_id, "model failed");
            error(StatusCode::BAD_GATEWAY, &ctx, "the model is unavailable")
        }
        Err(AgentError::Store(e)) => {
            tracing::error!(error = %e, request_id = %ctx.request_id, "store failed");
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                &ctx,
                "a store is unavailable",
            )
        }
    }
}

pub(crate) fn authorized(headers: &HeaderMap, token: &SecretString) -> bool {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|presented| presented == token.expose_secret())
}
