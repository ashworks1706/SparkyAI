//! Router assembly: /chat, the OpenAI-compatible surface, and /health.

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

pub mod chat;
pub mod health;
pub mod openai;
pub mod rate_limit;

/// Limits applied to the whole HTTP surface. Built from [http]; no Default.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Largest request body accepted, in bytes.
    pub max_body_bytes: usize,
    /// Requests handled at once. Zero removes the limit.
    pub concurrency: usize,
}

/// Cross-origin policy. An origins list holding a single * allows any origin; an empty list
/// adds no CORS headers.
pub fn cors(origins: &[String]) -> Option<CorsLayer> {
    if origins.is_empty() {
        return None;
    }
    if origins.iter().any(|o| o == "*") {
        return Some(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }
    let parsed: Vec<_> = origins.iter().filter_map(|o| o.parse().ok()).collect();
    Some(
        CorsLayer::new()
            .allow_origin(parsed)
            .allow_methods(Any)
            .allow_headers(Any),
    )
}

/// Full application router.
pub fn router(
    chat_state: chat::ChatState,
    health_state: health::HealthState,
    limits: Limits,
    cors_origins: &[String],
) -> Router {
    let health = Router::new()
        .route("/health/live", get(health::live))
        .route("/health/ready", get(health::ready))
        .with_state(health_state);
    let chat = Router::new()
        .route("/chat", post(chat::chat))
        .route("/chat/stream", post(chat::stream))
        .route("/confirm", post(chat::confirm))
        .route("/v1/chat/completions", post(openai::completions))
        .with_state(chat_state);
    let mut router = Router::new()
        .merge(health)
        .merge(chat)
        .route("/v1/models", get(openai::models))
        .layer(DefaultBodyLimit::max(limits.max_body_bytes))
        .layer(TraceLayer::new_for_http());
    if let Some(layer) = cors(cors_origins) {
        router = router.layer(layer);
    }
    if limits.concurrency > 0 {
        router = router.layer(tower::limit::ConcurrencyLimitLayer::new(limits.concurrency));
    }
    router
}
