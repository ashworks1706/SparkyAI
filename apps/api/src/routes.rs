//! Router assembly: /chat (Phase 3), /health, /admin.

use axum::{Router, routing::get};
use tower_http::trace::TraceLayer;

/// Full application router.
pub fn router() -> Router {
    Router::new()
        .route("/health/live", get(crate::health::live))
        .route("/health/ready", get(crate::health::ready))
        .nest("/admin", crate::admin::router())
        .layer(TraceLayer::new_for_http())
}
