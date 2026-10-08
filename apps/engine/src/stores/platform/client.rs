//! The HTTP client every platform adapter shares, holding the machine token.

use std::fmt::Write as _;
use std::time::Duration;

use chrono::{DateTime, NaiveDateTime, Utc};
use reqwest::{Method, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;
use serde_json::Value;
use url::Url;

/// Why a platform call failed. Carries the platform's error text, never the token.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    /// The request did not complete.
    #[error("platform unreachable: {0}")]
    Transport(String),
    /// The request ran past its budget.
    #[error("platform did not answer within {0:?}")]
    Timeout(Duration),
    /// The platform answered with an error status.
    #[error("platform answered {status}: {message}")]
    Status {
        /// HTTP status.
        status: u16,
        /// The error field of the body, or empty.
        message: String,
    },
    /// The body was not the shape expected.
    #[error("platform sent an unreadable body: {0}")]
    Body(String),
}

impl PlatformError {
    /// The HTTP status the platform answered with, when it answered.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Status { status, .. } => Some(*status),
            _ => None,
        }
    }
}

/// One request: method, path under the root, query pairs, JSON body, and its own budget.
pub(super) struct Call<'a> {
    /// HTTP method.
    pub method: Method,
    /// Path from the root, starting with a slash.
    pub path: &'a str,
    /// Query pairs.
    pub query: &'a [(&'a str, String)],
    /// JSON body.
    pub body: Option<&'a Value>,
    /// Budget for this call instead of the client default.
    pub timeout: Option<Duration>,
}

impl<'a> Call<'a> {
    /// A call with no query, body, or budget of its own.
    pub fn new(method: Method, path: &'a str) -> Self {
        Self {
            method,
            path,
            query: &[],
            body: None,
            timeout: None,
        }
    }

    /// Adds query pairs.
    pub fn query(self, query: &'a [(&'a str, String)]) -> Self {
        Self { query, ..self }
    }

    /// Adds a JSON body.
    pub fn body(self, body: &'a Value) -> Self {
        Self {
            body: Some(body),
            ..self
        }
    }

    /// Sets the budget of this call.
    pub fn timeout(self, timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
            ..self
        }
    }
}

/// An HTTP client for the platform API, holding the machine token. Cheap to clone.
#[derive(Clone)]
pub struct PlatformClient {
    http: reqwest::Client,
    root: Url,
    token: SecretString,
    timeout: Duration,
}

impl PlatformClient {
    /// Builds a client for the platform at root with token, each call bounded by timeout.
    pub fn new(root: &str, token: SecretString, timeout: Duration) -> Result<Self, PlatformError> {
        let root = Url::parse(root.trim().trim_end_matches('/'))
            .map_err(|e| PlatformError::Transport(format!("platform url: {e}")))?;
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| PlatformError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            root,
            token,
            timeout,
        })
    }

    /// The agents path of one member, the Discord user id, followed by rest.
    pub fn member(user: &str, rest: &str) -> String {
        format!("/api/agents/members/{}{rest}", segment(user))
    }

    /// The accounts path of one member followed by rest.
    pub(super) fn account(user: &str, rest: &str) -> String {
        format!("/api/accounts/members/{}{rest}", segment(user))
    }

    /// Whether the platform health route answers 2xx.
    pub async fn ready(&self) -> bool {
        let mut url = self.root.clone();
        url.set_path("/health");
        match self.http.get(url).send().await {
            Ok(response) => response.status().is_success(),
            Err(error) => {
                tracing::warn!(error = %error.without_url(), "readiness: platform did not answer");
                false
            }
        }
    }

    /// Sends one call. Returns the status and the JSON body, Null when there is none.
    async fn send(&self, call: &Call<'_>) -> Result<(StatusCode, Value), PlatformError> {
        let mut url = self.root.clone();
        url.set_path(call.path);
        if !call.query.is_empty() {
            url.query_pairs_mut().extend_pairs(call.query);
        }
        let budget = call.timeout.unwrap_or(self.timeout);
        let mut request = self
            .http
            .request(call.method.clone(), url)
            .bearer_auth(self.token.expose_secret())
            .timeout(budget);
        if let Some(body) = call.body {
            request = request.json(body);
        }
        let failed = |e: reqwest::Error| {
            if e.is_timeout() {
                PlatformError::Timeout(budget)
            } else {
                PlatformError::Transport(e.without_url().to_string())
            }
        };
        let response = request.send().await.map_err(failed)?;
        let status = response.status();
        let text = response.text().await.map_err(failed)?;
        if text.trim().is_empty() {
            return Ok((status, Value::Null));
        }
        match serde_json::from_str(&text) {
            Ok(value) => Ok((status, value)),
            Err(_) if !status.is_success() => Ok((status, Value::Null)),
            Err(e) => Err(PlatformError::Body(e.to_string())),
        }
    }

    /// Sends one call; any status outside 2xx is an error.
    pub(super) async fn call(&self, call: &Call<'_>) -> Result<Value, PlatformError> {
        let (status, value) = self.send(call).await?;
        if status.is_success() {
            return Ok(value);
        }
        let message = value
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        Err(PlatformError::Status {
            status: status.as_u16(),
            message,
        })
    }
}

/// Percent-encodes one path segment.
pub(super) fn segment(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(byte));
        } else {
            // Writing to a String cannot fail.
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

/// Reads a field, failing when it is missing or of another type.
pub(super) fn field<T: DeserializeOwned>(value: &Value, name: &str) -> Result<T, PlatformError> {
    let found = value
        .get(name)
        .cloned()
        .ok_or_else(|| PlatformError::Body(format!("{name} is missing")))?;
    serde_json::from_value(found).map_err(|e| PlatformError::Body(format!("{name}: {e}")))
}

/// A platform timestamp field: ISO 8601, read as UTC when it carries no offset.
pub(super) fn timestamp(value: &Value, name: &str) -> Result<DateTime<Utc>, PlatformError> {
    let text: String = field(value, name)?;
    parse_time(&text).ok_or_else(|| PlatformError::Body(format!("{name} is not a timestamp")))
}

/// An optional platform timestamp field.
pub(super) fn optional_timestamp(
    value: &Value,
    name: &str,
) -> Result<Option<DateTime<Utc>>, PlatformError> {
    match value.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => timestamp(value, name).map(Some),
    }
}

/// Parses ISO 8601; a time without an offset is UTC.
pub(super) fn parse_time(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text)
        .map(|t| t.with_timezone(&Utc))
        .ok()
        .or_else(|| {
            NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f")
                .ok()
                .map(|t| t.and_utc())
        })
}
