//! Liveness and readiness.

use axum::http::StatusCode;

pub async fn live() -> StatusCode {
    StatusCode::OK
}

/// Will check Postgres, Redis, Qdrant, and the model endpoint once adapters exist.
pub async fn ready() -> StatusCode {
    StatusCode::OK
}
