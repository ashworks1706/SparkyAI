//! Built-in tools and MCP-backed tools. Each declares a RiskClass.

pub mod knowledge_search;
pub mod mcp;
pub mod query_source;
pub mod sandbox;
pub mod skills;

/// The structured payload a tool hands back beside its text.
///
/// A value that will not serialize is a bug in the type. The call still answers, since the text
/// is what the model reads, and the trace says what was lost.
pub fn structured<T: serde::Serialize>(value: &T) -> Option<serde_json::Value> {
    match serde_json::to_value(value) {
        Ok(value) => Some(value),
        Err(error) => {
            tracing::error!(error = %error, "tool payload would not serialize");
            None
        }
    }
}
