//! Built-in tools and MCP-backed tools. Each declares a RiskClass.

pub mod canvas;
pub mod files;
pub mod gcal;
pub mod grant;
pub mod http;
pub mod knowledge;
pub mod mcp;
#[allow(
    dead_code,
    reason = "called by the per-user session routes of roadmap phase 8"
)]
pub mod oauth;
pub mod outlook;
pub mod papers;
pub mod sandbox;
pub mod transit;
pub mod wiki;

/// The structured payload a tool hands back beside its text; unserializable values log, yield None.
pub fn structured<T: serde::Serialize>(value: &T) -> Option<serde_json::Value> {
    match serde_json::to_value(value) {
        Ok(value) => Some(value),
        Err(error) => {
            tracing::error!(error = %error, "tool payload would not serialize");
            None
        }
    }
}
