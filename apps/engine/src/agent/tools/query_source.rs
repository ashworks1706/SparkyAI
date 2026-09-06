//! `ReadPublic`: run a live parameterized source query through the scraper's worker.
//!
//! One tool for every query source there will ever be. The sources come from the registry the
//! scraper publishes, so adding one is a scraper change and costs the model no extra schema.

use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::core::traits::query::SourceQueries;
use crate::core::traits::tool::Tool;
use crate::core::types::context::RequestContext;
use crate::core::types::query::{QueryError, QueryRequest, QuerySourceInfo};
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};

/// Runs one live source query per call.
pub struct QuerySourceTool {
    queries: Arc<dyn SourceQueries>,
    definition: ToolDefinition,
}

/// The description the model reads: what each source answers and what it takes.
///
/// Parameters are documented in prose rather than as a `oneOf` per source. A schema that
/// changes shape with one field's value is more correct and, for a small model, likelier to
/// produce nothing usable.
pub fn describe(sources: &[QuerySourceInfo]) -> String {
    let mut text = String::from(
        "Fetch live data from an ASU site that the knowledge base does not cover. Slower than \
         search_knowledge_base; use it only when the answer must be current. Sources:",
    );
    for source in sources {
        let _ = write!(text, "\n\n`{}` — {}", source.key, source.description.trim());
        if source.params.is_empty() {
            continue;
        }
        text.push_str("\n  params:");
        for p in &source.params {
            let _ = write!(
                text,
                "\n    {}{} — {}",
                p.name,
                if p.required { " (required)" } else { "" },
                p.description.trim()
            );
            if let Some(example) = &p.example {
                let _ = write!(text, " e.g. {example}");
            }
        }
    }
    text
}

impl QuerySourceTool {
    /// Builds the tool over the sources the registry currently offers. `timeout_secs` overrides
    /// the agent's default, since a live fetch takes far longer than a database read.
    pub fn new(
        queries: Arc<dyn SourceQueries>,
        sources: &[QuerySourceInfo],
        timeout_secs: u64,
    ) -> Self {
        let keys: Vec<&str> = sources.iter().map(|s| s.key.as_str()).collect();
        Self {
            queries,
            definition: ToolDefinition {
                name: "query_source".into(),
                description: describe(sources),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "enum": keys,
                            "description": "Which source to query."
                        },
                        "params": {
                            "type": "object",
                            "description": "Parameters for that source, as named above."
                        }
                    },
                    "required": ["source", "params"]
                }),
                risk: RiskClass::ReadPublic,
                sequential: false,
                timeout_secs: Some(timeout_secs),
            },
        }
    }
}

#[async_trait]
impl Tool for QuerySourceTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn call(&self, ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        let request: QueryRequest =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        if request.source.trim().is_empty() {
            return Err(ToolError::InvalidArguments("source is empty".into()));
        }
        let outcome = self.queries.run(ctx, &request).await.map_err(|e| match e {
            // A rejection is the model's to fix — a missing parameter, an unknown source — so
            // it comes back as text it can act on rather than as a failed run.
            QueryError::Rejected(reason) => ToolError::InvalidArguments(reason),
            QueryError::Cancelled => ToolError::Cancelled,
            QueryError::Timeout(_) => ToolError::Timeout,
            store @ QueryError::Store(_) => ToolError::Failed(store.to_string()),
        })?;
        Ok(ToolOutput {
            content: format!(
                "Live result from {} ({}):\n\n{}",
                outcome.source, outcome.url, outcome.text
            ),
            data: serde_json::to_value(&outcome).ok(),
        })
    }
}
