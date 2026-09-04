//! `BrowserTask`, `BrowserResult`, `SandboxError`. Phase 7.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A browser task for the sandbox worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserTask {
    /// What to do, in the worker's task protocol.
    pub instruction: String,
    /// Allowed domains for this task.
    pub allowed_domains: Vec<String>,
}

/// What the worker observed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserResult {
    /// Structured, size-limited observation.
    pub observation: Value,
    /// Whether the task stopped early (CAPTCHA, MFA, unexpected page).
    pub stopped_early: bool,
}

/// Sandbox failures.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// The worker is not configured.
    #[error("sandbox not available")]
    Unavailable,
    /// The worker failed the task.
    #[error("sandbox: {0}")]
    Failed(String),
}
