//! HTTP surface limits.

use serde::Deserialize;

/// HTTP surface limits.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Http {
    /// Largest request body accepted, in bytes.
    pub max_body_bytes: usize,
    /// Requests handled at once. 0 removes the limit.
    pub concurrency_limit: usize,
    /// Requests one user may start per minute. 0 removes the limit.
    pub rate_limit_per_min: u32,
    /// Origins allowed to call the engine from a browser. A single * allows any; empty adds
    /// no CORS headers.
    pub cors_origins: Vec<String>,
    /// How long in-flight requests get to finish after a shutdown signal.
    pub shutdown_grace_secs: u64,
}

impl Default for Http {
    fn default() -> Self {
        Self {
            max_body_bytes: 1 << 20,
            concurrency_limit: 0,
            rate_limit_per_min: 0,
            cors_origins: Vec::new(),
            shutdown_grace_secs: 10,
        }
    }
}
