//! Router assembly: /chat, /health, /admin.

use axum::Router;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

pub mod admin;
pub mod chat;
pub mod health;

/// Full application router.
pub fn router(chat_state: crate::core::types::adapters::ChatState) -> Router {
    Router::new()
        .route("/health/live", get(health::live))
        .route("/health/ready", get(health::ready))
        .route("/chat", post(chat::chat))
        .with_state(chat_state)
        .nest("/admin", admin::router())
        .layer(TraceLayer::new_for_http())
}
