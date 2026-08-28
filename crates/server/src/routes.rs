//! Router assembly.

use axum::{Router, routing::get};
use tower_http::trace::TraceLayer;

pub fn router() -> Router {
    Router::new()
        .route("/health/live", get(crate::health::live))
        .route("/health/ready", get(crate::health::ready))
        .nest("/admin", crate::admin::router())
        .layer(TraceLayer::new_for_http())
}
