//! Router assembly: /chat and /health.

use axum::Router;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

pub mod chat;
pub mod health;

/// Full application router.
pub fn router(chat_state: chat::ChatState, health_state: health::HealthState) -> Router {
    let health = Router::new()
        .route("/health/live", get(health::live))
        .route("/health/ready", get(health::ready))
        .with_state(health_state);
    let chat = Router::new()
        .route("/chat", post(chat::chat))
        .with_state(chat_state);
    Router::new()
        .merge(health)
        .merge(chat)
        .layer(TraceLayer::new_for_http())
}
