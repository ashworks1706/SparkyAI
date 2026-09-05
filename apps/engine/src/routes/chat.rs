//! `POST /chat`: one user message in, one answer with citations out. `POST /chat/stream` runs
//! the same turn and reports each step as it happens before the same answer.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::{StreamExt, stream};
use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use secrecy::{ExposeSecret, SecretString};
use serde_json::{Value, json};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::oneshot;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::agent::harness::agent::Agent;
use crate::core::traits::conversation::ConversationStore;
use crate::core::types::agent::AgentError;
use crate::core::types::chat::{ChatRequest, ChatResponse, ErrorBody};
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::model::ModelError;
use crate::core::types::wire::Progress;
use uuid::Uuid;

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
    match run_turn(state, req, None).instrument(span).await {
        Ok(answer) => Json(answer).into_response(),
        Err(failure) => failure.into_response(),
    }
}

/// The same turn as `chat`, reported as it happens: a `progress` event per step worth showing,
/// then one `answer` event with the usual `ChatResponse` or an error body, then `done`.
pub async fn stream(
    State(state): State<ChatState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    let span = tracing::info_span!("http.chat.stream", "sparky.user_id" = %req.user_id);
    if let Some(parent) = headers
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_traceparent)
        && let Err(e) = span.set_parent(parent)
    {
        tracing::debug!(error = %e, "traceparent ignored");
    }

    let (progress_tx, progress_rx) = mpsc::unbounded_channel();
    let (answer_tx, answer_rx) = oneshot::channel();
    tokio::spawn(
        async move {
            let _ = answer_tx.send(run_turn(state, req, Some(progress_tx)).await);
        }
        .instrument(span),
    );

    let progress = UnboundedReceiverStream::new(progress_rx)
        .map(|p| sse("progress", &serde_json::to_value(&p).unwrap_or_default()));
    let tail = stream::once(async move {
        match answer_rx.await {
            Ok(Ok(answer)) => sse("answer", &serde_json::to_value(&answer).unwrap_or_default()),
            Ok(Err(failure)) => sse(
                "error",
                &serde_json::to_value(&failure.body).unwrap_or_default(),
            ),
            Err(e) => sse("error", &json!({ "error": e.to_string() })),
        }
    })
    .chain(stream::once(async { sse("done", &json!({})) }));

    Sse::new(progress.chain(tail).map(Ok::<Event, Infallible>)).into_response()
}

fn sse(name: &str, body: &Value) -> Event {
    Event::default().event(name).data(body.to_string())
}

/// One turn, with the progress channel the caller wants events on, if any.
async fn run_turn(
    state: ChatState,
    req: ChatRequest,
    watcher: Option<UnboundedSender<Progress>>,
) -> Result<ChatResponse, Failure> {
    if req.message.trim().is_empty() {
        return Err(Failure::new(
            StatusCode::BAD_REQUEST,
            Uuid::nil(),
            "message is empty",
        ));
    }
    let tenant = req
        .tenant_id
        .unwrap_or_else(|| state.default_tenant.clone());
    let mut ctx =
        RequestContext::new(tenant, req.user_id, state.request_budget).with_roles(req.roles);
    if let Some(id) = req.conversation_id {
        ctx = ctx.with_conversation(id);
    }
    if let Some(tx) = watcher {
        ctx = ctx.listening_to(tx);
    }
    let id = ctx.request_id;
    if let Some(store) = &state.conversations
        && let Err(e) = store.ensure(&ctx, &req.channel_id).await
    {
        tracing::error!(error = %e, "conversation store unavailable");
        return Err(Failure::new(
            StatusCode::SERVICE_UNAVAILABLE,
            id,
            "conversation store unavailable",
        ));
    }
    match state.agent.run(&ctx, &req.message).await {
        Ok(answer) => Ok(ChatResponse {
            request_id: id,
            conversation_id: ctx.conversation_id,
            citations: answer.evidence.iter().map(Evidence::citation).collect(),
            text: answer.text,
            confirmation: answer.confirmation,
            status: answer.status,
            steps: answer.steps,
            tools: answer.tool_runs,
            tokens: answer.usage.total(),
            cost_usd: answer.cost_usd,
        }),
        Err(AgentError::Model(ModelError::Busy)) => {
            tracing::warn!(request_id = %id, "model at capacity");
            Err(Failure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                id,
                "the model is at capacity",
            ))
        }
        Err(AgentError::Model(e)) => {
            tracing::error!(error = %e, request_id = %id, "model failed");
            Err(Failure::new(
                StatusCode::BAD_GATEWAY,
                id,
                "the model is unavailable",
            ))
        }
        Err(AgentError::Store(e)) => {
            tracing::error!(error = %e, request_id = %id, "store failed");
            Err(Failure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                id,
                "a store is unavailable",
            ))
        }
    }
}

/// A turn that could not produce an answer.
struct Failure {
    status: StatusCode,
    body: ErrorBody,
}

impl Failure {
    fn new(status: StatusCode, request_id: Uuid, error: &str) -> Self {
        Self {
            status,
            body: ErrorBody {
                request_id,
                error: error.to_owned(),
            },
        }
    }
}

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

pub(crate) fn authorized(headers: &HeaderMap, token: &SecretString) -> bool {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|presented| presented == token.expose_secret())
}
