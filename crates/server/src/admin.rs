//! Tools, sources, instructions, limits, traces, approvals, rollback. Phase 4.

use axum::Router;

/// Admin routes, mounted under `/admin`.
pub fn router() -> Router {
    Router::new()
}
