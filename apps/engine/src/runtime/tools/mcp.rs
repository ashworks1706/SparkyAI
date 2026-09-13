//! MCP servers as tools. Each remote tool becomes a Tool with a RiskClass derived from its
//! name, gated by Policy like any built-in.

use std::sync::Arc;

use async_trait::async_trait;
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::service::{Peer, RoleClient};
use rmcp::transport::StreamableHttpClientTransport;
use serde_json::Value;

use crate::core::traits::tools::Tool;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};

/// Limits applied to the tools of one MCP server.
#[derive(Debug, Clone)]
pub struct McpLimits {
    /// Longest tool result handed back to the model.
    pub max_output_chars: usize,
    /// Longest per-property description kept in a schema.
    pub max_schema_description_chars: usize,
    /// Longest tool description kept.
    pub max_tool_description_chars: usize,
    /// Show the model only the required properties of each tool.
    pub required_props_only: bool,
    /// Per-tool timeout for this server, overriding the default of the agent.
    pub tool_timeout_secs: Option<u64>,
}

impl Default for McpLimits {
    fn default() -> Self {
        Self::from(&crate::core::config::Mcp::default())
    }
}

impl From<&crate::core::config::Mcp> for McpLimits {
    /// Server-level limits, before the overrides of a single server.
    fn from(cfg: &crate::core::config::Mcp) -> Self {
        Self {
            max_output_chars: cfg.max_output_chars,
            max_schema_description_chars: cfg.max_schema_description_chars,
            max_tool_description_chars: cfg.max_tool_description_chars,
            required_props_only: cfg.required_props_only,
            tool_timeout_secs: None,
        }
    }
}

/// Keeps only the required properties of an object schema.
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

/// Drops schema noise the model does not need: long descriptions, titles, examples, $schema.
pub fn compact_schema(value: Value, max_description: usize) -> Value {
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
                        return (k, Value::String(s.chars().take(max_description).collect()));
                    }
                    (k, compact_schema(v, max_description))
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|v| compact_schema(v, max_description))
                .collect(),
        ),
        other => other,
    }
}

/// One remote MCP tool, callable through the harness.
pub struct McpTool {
    peer: Peer<RoleClient>,
    definition: ToolDefinition,
    max_output_chars: usize,
}

/// Risk by name. Reads and inspection run, interactions are drafts, and anything that submits
/// or is unrecognised must be confirmed.
pub fn risk_for(name: &str) -> RiskClass {
    /// Look at the page without changing it.
    const READS: [&str; 10] = [
        "navigate",
        "snapshot",
        "screenshot",
        "find",
        "tabs",
        "wait_for",
        "console",
        "network_requests",
        "resize",
        "install",
    ];
    /// Change what the page holds without committing it.
    const DRAFTS: [&str; 5] = ["type", "fill_form", "select_option", "hover", "drag"];
    /// Can commit the page or run code in it. Listed by name.
    const COMMITS: [&str; 6] = [
        "submit",
        "click",
        "press_key",
        "evaluate",
        "file_upload",
        "handle_dialog",
    ];
    if COMMITS.iter().any(|k| name.contains(k)) {
        return RiskClass::ExternalWrite;
    }
    if DRAFTS.iter().any(|k| name.contains(k)) {
        return RiskClass::PrepareWrite;
    }
    if READS.iter().any(|k| name.contains(k)) {
        return RiskClass::ReadPublic;
    }
    // An unrecognised tool is treated as an external write.
    RiskClass::ExternalWrite
}

/// Connects to a Streamable-HTTP MCP server and wraps its tools. allow limits which remote
/// tools are exposed, and empty means all. The connection lives as long as the process.
pub async fn connect(
    url: &str,
    allow: &[String],
    limits: &McpLimits,
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
                .take(limits.max_tool_description_chars)
                .collect(),
            parameters: {
                let schema = compact_schema(
                    Value::Object((*t.input_schema).clone()),
                    limits.max_schema_description_chars,
                );
                if limits.required_props_only {
                    required_only(schema)
                } else {
                    schema
                }
            },
            name,
            sequential: true,
            timeout_secs: limits.tool_timeout_secs,
        };
        tools.push(Arc::new(McpTool {
            peer: peer.clone(),
            definition,
            max_output_chars: limits.max_output_chars,
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
        if text.chars().count() > self.max_output_chars {
            let cut: String = text.chars().take(self.max_output_chars).collect();
            text = format!("{cut}\n…[truncated]");
        }
        Ok(ToolOutput {
            content: text,
            data: result.structured_content,
            sources: Vec::new(),
        })
    }
}
