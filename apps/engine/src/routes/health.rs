//! Liveness and readiness.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use sqlx::postgres::PgPool;

use crate::core::types::http::health::Readiness;

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

/// Returns 200 when Postgres and the model endpoint both answer, 503 with the report otherwise.
pub async fn ready(State(state): State<HealthState>) -> Response {
    let postgres = match sqlx::query("select 1").execute(&state.pool).await {
        Ok(_) => true,
        Err(error) => {
            tracing::warn!(%error, "readiness: postgres did not answer");
            false
        }
    };
    let model = match reqwest::Client::new()
        .get(format!(
            "{}/models",
            state.model_base_url.trim_end_matches('/')
        ))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => true,
        Ok(r) => {
            tracing::warn!(status = %r.status(), "readiness: model endpoint refused");
            false
        }
        Err(error) => {
            tracing::warn!(%error, "readiness: model endpoint did not answer");
            false
        }
    };
    let report = Readiness { postgres, model };
    let status = if postgres && model {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(report)).into_response()
}
