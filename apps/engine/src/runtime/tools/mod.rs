//! Built-in tools and MCP-backed tools. Each declares a RiskClass.

pub mod knowledge;
pub mod mcp;
pub mod sandbox;

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
