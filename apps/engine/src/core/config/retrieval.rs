//! Settings that shape what is fetched and what the engine serves.

use serde::Deserialize;

/// Hybrid retrieval tuning. Dense and lexical are fused with reciprocal rank fusion.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Retrieval {
    /// Evidence chunks handed to the prompt per request.
    pub top_k: usize,
    /// Candidates pulled from each leg before fusion.
    pub candidates: i64,
    /// Reciprocal rank fusion constant. Lower trusts the top of each list more.
    pub rrf_k: f32,
    /// PostgreSQL text search configuration for the lexical leg.
    pub text_search_config: String,
    /// Run the pgvector leg.
    pub dense: bool,
    /// Run the full-text leg.
    pub lexical: bool,
    /// Drop fused results below this score. Zero keeps everything.
    pub min_score: f32,
    /// Drop a chunk when the summary covering it is already in the result.
    pub collapse_tree: bool,
}

impl Default for Retrieval {
    fn default() -> Self {
        Self {
            top_k: 6,
            candidates: 20,
            rrf_k: 60.0,
            text_search_config: "english".into(),
            dense: true,
            lexical: true,
            min_score: 0.0,
            collapse_tree: true,
        }
    }
}

/// MCP servers exposed as tools.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Mcp {
    /// Servers to connect to at boot. Expressed in the TOML layer.
    pub servers: Vec<McpServer>,
    /// Default for a server that does not set required_props_only.
    pub required_props_only: bool,
    /// Longest tool result handed back to the model.
    pub max_output_chars: usize,
    /// Longest per-property description kept in a tool schema.
    pub max_schema_description_chars: usize,
    /// Longest tool description kept.
    pub max_tool_description_chars: usize,
    /// Legacy single-server form, folded into servers as playwright. Prefer servers.
    pub playwright_url: Option<String>,
    /// Tools exposed by the legacy playwright_url server.
    pub playwright_tools: Vec<String>,
}

/// One MCP server.
#[derive(Debug, Clone, Deserialize)]
pub struct McpServer {
    /// Name used in logs and errors.
    pub name: String,
    /// Streamable-HTTP endpoint, e.g. http://localhost:8931/mcp.
    pub url: String,
    /// Remote tool names to expose; empty exposes every tool the server lists.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Overrides mcp.required_props_only for this server.
    #[serde(default)]
    pub required_props_only: Option<bool>,
    /// Overrides agent.tool_timeout_secs for the tools of this server.
    #[serde(default)]
    pub tool_timeout_secs: Option<u64>,
}

impl Mcp {
    /// Configured servers, with the legacy playwright_url folded in and empty URLs dropped.
    pub fn resolved_servers(&self) -> Vec<McpServer> {
        let mut out: Vec<McpServer> = self
            .servers
            .iter()
            .filter(|s| !s.url.trim().is_empty())
            .cloned()
            .collect();
        if let Some(url) = self
            .playwright_url
            .as_deref()
            .map(str::trim)
            .filter(|u| !u.is_empty())
            && !out.iter().any(|s| s.name == "playwright")
        {
            out.push(McpServer {
                name: "playwright".into(),
                url: url.to_owned(),
                tools: self.playwright_tools.clone(),
                required_props_only: None,
                tool_timeout_secs: None,
            });
        }
        out
    }
}

impl Default for Mcp {
    fn default() -> Self {
        Self {
            servers: Vec::new(),
            required_props_only: true,
            max_output_chars: 6_000,
            max_schema_description_chars: 80,
            max_tool_description_chars: 160,
            playwright_url: None,
            playwright_tools: [
                "browser_navigate",
                "browser_navigate_back",
                "browser_snapshot",
                "browser_click",
                "browser_type",
                "browser_press_key",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        }
    }
}

/// JSONL trace recording. One file per request under dir.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Trace {
    /// Write traces at all.
    pub enabled: bool,
    /// Directory for JSONL traces.
    pub dir: String,
    /// Stop writing the trace of a request past this many bytes. 0 removes the limit.
    pub max_file_bytes: u64,
    /// Delete traces older than this at boot. 0 keeps them forever.
    pub retention_hours: u64,
}

impl Default for Trace {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: ".sparky/traces".into(),
            max_file_bytes: 0,
            retention_hours: 0,
        }
    }
}

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
