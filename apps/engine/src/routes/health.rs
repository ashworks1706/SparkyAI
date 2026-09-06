//! Liveness and readiness.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use sqlx::postgres::PgPool;

use crate::core::types::health::Readiness;

/// What readiness checks.
#[derive(Clone)]
pub struct HealthState {
    /// The store every request needs.
    pub pool: PgPool,
    /// Chat model base URL, ending in /v1.
    pub model_base_url: String,
}

/// Process is up.
pub async fn live() -> StatusCode {
    StatusCode::OK
}

/// Returns 200 when Postgres and the model endpoint both answer, and 503 with the report when
/// either does not.
pub async fn ready(State(state): State<HealthState>) -> Response {
    let postgres = sqlx::query("select 1").execute(&state.pool).await.is_ok();
    let model = reqwest::Client::new()
        .get(format!(
            "{}/models",
            state.model_base_url.trim_end_matches('/')
        ))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .is_ok_and(|r| r.status().is_success());
    let report = Readiness { postgres, model };
    let status = if postgres && model {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(report)).into_response()
}
