//! Failure: the error response every route returns, and the mapping from an agent error to it.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use uuid::Uuid;

use crate::core::types::agent::AgentError;
use crate::core::types::http::chat::ErrorBody;
use crate::core::types::model::ModelError;

/// What a caller hears about a conversation that is not theirs, whether or not it exists.
pub const NO_SUCH_CONVERSATION: &str = "no such conversation";

/// A turn that could not produce an answer.
pub(crate) struct Failure {
    status: StatusCode,
    body: ErrorBody,
}

impl Failure {
    /// A failure with this status and message for the request.
    pub(crate) fn new(status: StatusCode, request_id: Uuid, error: &str) -> Self {
        Self {
            status,
            body: ErrorBody {
                request_id,
                error: error.to_owned(),
                status: Some(status.as_u16()),
            },
        }
    }

    /// The failure an agent error maps to, logged with the request it ended.
    pub(crate) fn from_agent(request_id: Uuid, error: AgentError) -> Self {
        match error {
            AgentError::Model(ModelError::Busy) => {
                tracing::warn!(request_id = %request_id, "model at capacity");
                Self::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    request_id,
                    "the model is at capacity",
                )
            }
            AgentError::Model(e) => {
                tracing::error!(error = %e, request_id = %request_id, "model failed");
                Self::new(
                    StatusCode::BAD_GATEWAY,
                    request_id,
                    "the model is unavailable",
                )
            }
            AgentError::Store(e) => {
                tracing::error!(error = %e, request_id = %request_id, "store failed");
                Self::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    request_id,
                    "a store is unavailable",
                )
            }
        }
    }

    /// The body sent to the caller.
    pub(crate) fn body(&self) -> &ErrorBody {
        &self.body
    }

    /// The status with the message as plain text, for routes that answer without a JSON body.
    pub(crate) fn into_text(self) -> Response {
        (self.status, self.body.error).into_response()
    }
}

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}
