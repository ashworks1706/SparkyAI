//! MCP servers as tools. Each remote tool becomes a `Tool` with a `RiskClass` derived from its
//! name, so `Policy` gates it like any built-in. Today this serves Playwright MCP; Phase 7 puts
//! the browser behind per-user contexts and tightens the risk mapping.

use std::sync::Arc;

use async_trait::async_trait;
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::service::{Peer, RoleClient};
use rmcp::transport::StreamableHttpClientTransport;
use serde_json::Value;

use crate::core::traits::tool::Tool;
use crate::core::types::context::RequestContext;
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};

/// Longest tool result handed back to the model; page snapshots can be enormous.
const MAX_OUTPUT_CHARS: usize = 6_000;
/// Longest per-property description kept in a schema. Tool schemas count against the
/// context window on every step, so verbose ones are trimmed.
const MAX_SCHEMA_DESCRIPTION: usize = 80;

/// Keeps only the `required` properties of an object schema. Small models tend to fill every
/// optional field they are shown, which on browser tools means wrong targets and snapshots
/// written to files instead of returned.
pub fn required_only(value: Value) -> Value {
    let Value::Object(mut map) = value else {
        return value;
    };
    let required: Vec<String> = map
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    if let Some(Value::Object(props)) = map.get_mut("properties") {
        props.retain(|k, _| required.contains(k));
    }
    Value::Object(map)
}

/// Drops schema noise the model does not need: long descriptions, titles, examples, `$schema`.
pub fn compact_schema(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(k, _)| {
                    !matches!(k.as_str(), "title" | "examples" | "$schema" | "default")
                })
                .map(|(k, v)| {
                    if k == "description"
                        && let Value::String(s) = &v
                    {
                        return (
                            k,
                            Value::String(s.chars().take(MAX_SCHEMA_DESCRIPTION).collect()),
                        );
                    }
                    (k, compact_schema(v))
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(compact_schema).collect()),
        other => other,
    }
}

/// One remote MCP tool, callable through the harness.
pub struct McpTool {
    peer: Peer<RoleClient>,
    definition: ToolDefinition,
}

/// Risk by name. Reads and inspection run; interactions are drafts; anything that submits or
/// is unrecognised must be confirmed.
pub fn risk_for(name: &str) -> RiskClass {
    const READS: [&str; 10] = [
        "navigate",
        "snapshot",
        "screenshot",
        "find",
        "tabs",
        "wait_for",
        "console",
        "network_requests",
        "evaluate",
        "resize",
    ];
    const DRAFTS: [&str; 9] = [
        "click",
        "type",
        "fill_form",
        "select_option",
        "press_key",
        "hover",
        "drag",
        "file_upload",
        "handle_dialog",
    ];
    if name.contains("submit") {
        return RiskClass::ExternalWrite;
    }
    if READS.iter().any(|k| name.contains(k)) {
        return RiskClass::ReadPublic;
    }
    if DRAFTS.iter().any(|k| name.contains(k)) {
        return RiskClass::PrepareWrite;
    }
    RiskClass::ExternalWrite
}

/// Connects to a Streamable-HTTP MCP server and wraps its tools. `allow` limits which remote
/// tools are exposed; empty means all. The connection lives as long as the process.
pub async fn connect(
    url: &str,
    allow: &[String],
    required_props_only: bool,
) -> Result<Vec<Arc<dyn Tool>>, String> {
    let transport = StreamableHttpClientTransport::from_uri(url);
    let service = ().serve(transport).await.map_err(|e| e.to_string())?;
    let peer = service.peer().clone();
    tokio::spawn(async move {
        if let Err(e) = service.waiting().await {
            tracing::warn!(error = %e, "mcp connection ended");
        }
    });
    let remote = peer.list_all_tools().await.map_err(|e| e.to_string())?;
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    for t in remote {
        let name = t.name.to_string();
        if !allow.is_empty() && !allow.iter().any(|a| a == &name) {
            continue;
        }
        let definition = ToolDefinition {
            risk: risk_for(&name),
            description: t
                .description
                .as_deref()
                .unwrap_or(&name)
                .chars()
                .take(160)
                .collect(),
            parameters: {
                let schema = compact_schema(Value::Object((*t.input_schema).clone()));
                if required_props_only {
                    required_only(schema)
                } else {
                    schema
                }
            },
            name,
            sequential: true,
        };
        tools.push(Arc::new(McpTool {
            peer: peer.clone(),
            definition,
        }));
    }
    Ok(tools)
}

#[async_trait]
impl Tool for McpTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn call(&self, _ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        let arguments = match args {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => {
                return Err(ToolError::InvalidArguments(format!(
                    "expected an object, got {other}"
                )));
            }
        };
        let mut params = CallToolRequestParams::new(self.definition.name.clone());
        params.arguments = arguments;
        let result = self
            .peer
            .call_tool(params)
            .await
            .map_err(|e| ToolError::Failed(e.to_string()))?;
        let mut text = String::new();
        for block in &result.content {
            if let Some(t) = block.as_text() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&t.text);
            }
        }
        if result.is_error.unwrap_or(false) {
            return Err(ToolError::Failed(if text.is_empty() {
                "mcp tool reported an error".into()
            } else {
                text
            }));
        }
        if text.chars().count() > MAX_OUTPUT_CHARS {
            let cut: String = text.chars().take(MAX_OUTPUT_CHARS).collect();
            text = format!("{cut}\n…[truncated]");
        }
        Ok(ToolOutput {
            content: text,
            data: result.structured_content,
        })
    }
}
