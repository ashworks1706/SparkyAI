//! POST /chat takes one user message and returns one answer with citations. POST /chat/stream
//! runs the same turn and reports each step as it happens before the same answer.

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
use serde::Serialize;
use serde_json::json;
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::oneshot;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::Instrument;
use tracing::field::Empty;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::core::traits::conversation::ConversationStore;
use crate::core::traits::safety::confirmation::ConfirmationStore;
use crate::core::types::agent::AgentError;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::http::chat::{ChatRequest, ChatResponse, ConfirmRequest, ErrorBody};
use crate::core::types::model::ModelError;
use crate::core::types::store::StoreError;
use crate::core::types::trace::progress::Progress;
use crate::routes::rate_limit::RateLimiter;
use crate::runtime::harness::agent::Agent;
use uuid::Uuid;

/// What the chat route needs.
#[derive(Clone)]
pub struct ChatState {
    /// The agent.
    pub agent: Agent,
    /// To create the conversation row before the run.
    pub conversations: Option<Arc<dyn ConversationStore>>,
    /// Where actions wait for approval.
    pub confirmations: Option<Arc<dyn ConfirmationStore>>,
    /// Per-request wall-clock budget.
    pub request_budget: Duration,
    /// Tenant used when the client sends none (single-guild deployments).
    pub default_tenant: String,
    /// Bearer token every caller must present.
    pub service_token: SecretString,
    /// Per-user request limit.
    pub rate_limit: RateLimiter,
}

/// What a caller hears about a conversation that is not theirs, whether or not it exists.
pub const NO_SUCH_CONVERSATION: &str = "no such conversation";

/// The 429 returned when a caller is over the limit.
pub fn too_many(user: &str) -> Response {
    tracing::warn!(user, "rate limited");
    (
        StatusCode::TOO_MANY_REQUESTS,
        "too many requests; wait a minute and ask again",
    )
        .into_response()
}

/// Parses a W3C traceparent header into a remote parent context.
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

/// The span one request runs under, carrying the attributes both trace UIs read. The answer
/// and the outcome are recorded on it when the turn ends.
macro_rules! route_span {
    ($name:literal, $req:expr) => {
        tracing::info_span!(
            $name,
            "posthog.distinct_id" = %$req.user_id,
            "openinference.span.kind" = "CHAIN",
            "input.value" = %$req.message,
            "output.value" = Empty,
            "user.id" = %$req.user_id,
            "session.id" = Empty,
            "otel.status_code" = Empty,
            "otel.status_message" = Empty,
        )
    };
}

/// Records how a turn ended on the span it ran under.
fn record_outcome(span: &tracing::Span, outcome: &Result<ChatResponse, Failure>) {
    match outcome {
        Ok(answer) => {
            span.record("session.id", answer.conversation_id.to_string().as_str());
            span.record(
                "output.value",
                answer.text.chars().take(4_000).collect::<String>().as_str(),
            );
            span.record("otel.status_code", "OK");
        }
        Err(failure) => {
            span.record("otel.status_code", "ERROR");
            span.record("otel.status_message", failure.body.error.as_str());
        }
    }
}

/// Handles one chat turn. Runs under an http.chat span parented to the traceparent of the
/// caller.
pub async fn chat(
    State(state): State<ChatState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    if !state.rate_limit.allow(&req.user_id) {
        return too_many(&req.user_id);
    }
    let span = route_span!("http.chat", req);
    if let Some(parent) = headers
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_traceparent)
        && let Err(e) = span.set_parent(parent)
    {
        tracing::debug!(error = %e, "traceparent ignored");
    }
    let outcome = run_turn(state, req, None).instrument(span.clone()).await;
    record_outcome(&span, &outcome);
    match outcome {
        Ok(answer) => Json(answer).into_response(),
        Err(failure) => failure.into_response(),
    }
}

/// The same turn as chat, reported as it happens. A progress event per step worth showing,
/// then one answer event with a ChatResponse or an error body, then done.
pub async fn stream(
    State(state): State<ChatState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    if !state.rate_limit.allow(&req.user_id) {
        return too_many(&req.user_id);
    }
    let span = route_span!("http.chat.stream", req);
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
    let outcome_span = span.clone();
    tokio::spawn(
        async move {
            let outcome = run_turn(state, req, Some(progress_tx)).await;
            record_outcome(&outcome_span, &outcome);
            let _ = answer_tx.send(outcome);
        }
        .instrument(span),
    );

    let progress = UnboundedReceiverStream::new(progress_rx).map(|p| sse("progress", &p));
    let tail = stream::once(async move {
        match answer_rx.await {
            Ok(Ok(answer)) => sse("answer", &answer),
            Ok(Err(failure)) => sse("error", &failure.body),
            Err(e) => sse("error", &json!({ "error": e.to_string() })),
        }
    })
    .chain(stream::once(async { sse("done", &json!({})) }));

    Sse::new(progress.chain(tail).map(Ok::<Event, Infallible>)).into_response()
}

/// One server-sent event. A payload that cannot be serialised is reported to the client as an
/// error and logged.
fn sse<T: Serialize>(name: &str, body: &T) -> Event {
    match serde_json::to_string(body) {
        Ok(json) => Event::default().event(name).data(json),
        Err(e) => {
            tracing::error!(error = %e, event = name, "could not serialise a stream event");
            Event::default()
                .event("error")
                .data(json!({ "error": "internal serialisation failure" }).to_string())
        }
    }
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
    let ctx = RequestContext::new(tenant, req.user_id, state.request_budget)
        .with_roles(req.roles)
        .with_visibility(req.visibility);
    let mut ctx = open(
        &state,
        ctx,
        req.conversation_id,
        req.continue_channel,
        &req.channel_id,
    )
    .await?;
    if let Some(tx) = watcher {
        ctx = ctx.listening_to(tx);
    }
    let id = ctx.request_id;
    match state.agent.run(&ctx, &req.message).await {
        Ok(answer) => Ok(ChatResponse {
            request_id: id,
            conversation_id: ctx.conversation_id,
            citations: answer.citations(),
            text: answer.text,
            confirmation: answer.confirmation,
            status: answer.status,
            steps: answer.steps,
            tools: answer.tool_runs,
            memories: answer.memories,
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

/// Picks the conversation a turn continues and ensures the caller owns it. A given id is
/// continued; otherwise continue_channel picks the newest open conversation of the caller in
/// the channel at the request visibility; otherwise the turn starts a new one.
async fn open(
    state: &ChatState,
    mut ctx: RequestContext,
    conversation_id: Option<Uuid>,
    continue_channel: bool,
    channel_id: &str,
) -> Result<RequestContext, Failure> {
    let Some(store) = &state.conversations else {
        if let Some(id) = conversation_id {
            ctx = ctx.with_conversation(id);
        }
        return Ok(ctx);
    };
    if let Some(id) = conversation_id {
        ctx = ctx.with_conversation(id);
    } else if continue_channel {
        match store.latest(&ctx, channel_id).await {
            Ok(Some(id)) => ctx = ctx.with_conversation(id),
            Ok(None) => {}
            Err(e) => {
                tracing::error!(error = %e, "conversation store unavailable");
                return Err(Failure::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    ctx.request_id,
                    "conversation store unavailable",
                ));
            }
        }
    }
    match store.ensure(&ctx, channel_id).await {
        Ok(()) => Ok(ctx),
        Err(StoreError::NotOwned) => {
            tracing::warn!(user = %ctx.user_id, "conversation held by another caller");
            Err(Failure::new(
                StatusCode::NOT_FOUND,
                ctx.request_id,
                NO_SUCH_CONVERSATION,
            ))
        }
        Err(e) => {
            tracing::error!(error = %e, "conversation store unavailable");
            Err(Failure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                ctx.request_id,
                "conversation store unavailable",
            ))
        }
    }
}

/// Answers a held action. Approving runs it and lets the agent finish; denying drops it.
///
/// A token is single use and belongs to the caller who was asked, both enforced by the store.
pub async fn confirm(
    State(state): State<ChatState>,
    headers: HeaderMap,
    Json(req): Json<ConfirmRequest>,
) -> Response {
    if !authorized(&headers, &state.service_token) {
        return (StatusCode::UNAUTHORIZED, "missing or wrong bearer token").into_response();
    }
    let Some(store) = &state.confirmations else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "this engine holds no confirmations",
        )
            .into_response();
    };
    let tenant = req
        .tenant_id
        .unwrap_or_else(|| state.default_tenant.clone());
    let ctx = RequestContext::new(tenant, req.user_id, state.request_budget)
        .with_conversation(req.conversation_id)
        .with_visibility(req.visibility);
    if let Some(conversations) = &state.conversations {
        match conversations.owns(&ctx).await {
            Ok(true) => {}
            Ok(false) => {
                return Failure::new(StatusCode::NOT_FOUND, ctx.request_id, NO_SUCH_CONVERSATION)
                    .into_response();
            }
            Err(e) => {
                tracing::error!(error = %e, "conversation store failed");
                return Failure::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    ctx.request_id,
                    "a store is unavailable",
                )
                .into_response();
            }
        }
    }

    let claimed = match store.claim(&ctx, req.token, req.approve).await {
        Ok(claimed) => claimed,
        Err(e) => {
            tracing::error!(error = %e, "confirmation store failed");
            return Failure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                ctx.request_id,
                "a store is unavailable",
            )
            .into_response();
        }
    };
    let Some(pending) = claimed else {
        // Expired, already answered, or held for another caller.
        return Failure::new(
            StatusCode::CONFLICT,
            ctx.request_id,
            "that approval is no longer open",
        )
        .into_response();
    };
    if !req.approve {
        return Json(json!({
            "request_id": ctx.request_id,
            "conversation_id": ctx.conversation_id,
            "status": "denied",
            "text": "Fine, I will not do that.",
        }))
        .into_response();
    }
    match state.agent.resume(&ctx, pending).await {
        Ok(answer) => Json(ChatResponse {
            request_id: ctx.request_id,
            conversation_id: ctx.conversation_id,
            citations: answer.citations(),
            text: answer.text,
            confirmation: answer.confirmation,
            status: answer.status,
            steps: answer.steps,
            tools: answer.tool_runs,
            memories: answer.memories,
            tokens: answer.usage.total(),
            cost_usd: answer.cost_usd,
        })
        .into_response(),
        Err(AgentError::Model(e)) => {
            tracing::error!(error = %e, "model failed after approval");
            Failure::new(
                StatusCode::BAD_GATEWAY,
                ctx.request_id,
                "the model is unavailable",
            )
            .into_response()
        }
        Err(AgentError::Store(e)) => {
            tracing::error!(error = %e, "store failed after approval");
            Failure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                ctx.request_id,
                "a store is unavailable",
            )
            .into_response()
        }
    }
}

/// A turn that could not produce an answer.
pub(crate) struct Failure {
    status: StatusCode,
    body: ErrorBody,
}

impl Failure {
    pub(crate) fn new(status: StatusCode, request_id: Uuid, error: &str) -> Self {
        Self {
            status,
            body: ErrorBody {
                request_id,
                error: error.to_owned(),
                status: Some(status.as_u16()),
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
