//! Router assembly: /chat (Phase 3), /health, /admin.

use axum::{Router, routing::get};
use tower_http::trace::TraceLayer;

pub mod admin;
pub mod health;

/// Full application router.
pub fn router() -> Router {
    Router::new()
        .route("/health/live", get(health::live))
        .route("/health/ready", get(health::ready))
        .nest("/admin", admin::router())
        .layer(TraceLayer::new_for_http())
}
