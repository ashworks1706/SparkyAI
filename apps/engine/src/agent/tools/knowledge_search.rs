//! ReadPublic: search the indexed knowledge base through Retriever.

use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::tools::structured;
use crate::core::traits::retrieval::Retriever;
use crate::core::traits::tool::Tool;
use crate::core::types::context::RequestContext;
use crate::core::types::retrieval::RetrievalQuery;
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::core::types::wire::SearchArgs;

/// Runs a targeted search over the indexed knowledge base.
pub struct KnowledgeSearch {
    retriever: Arc<dyn Retriever>,
    top_k: usize,
    /// The categories the index holds, sorted. Named in the schema and checked on every call.
    categories: Vec<String>,
}

impl KnowledgeSearch {
    /// Searches with retriever, returning at most top_k chunks. categories are the ones the
    /// scraper has published.
    pub fn new(retriever: Arc<dyn Retriever>, top_k: usize, categories: Vec<String>) -> Self {
        Self {
            retriever,
            top_k,
            categories,
        }
    }

    /// The categories asked for that the index does not hold.
    fn unknown(&self, asked: &[String]) -> Vec<String> {
        asked
            .iter()
            .filter(|c| !self.categories.iter().any(|k| k.eq_ignore_ascii_case(c)))
            .cloned()
            .collect()
    }
}

#[async_trait]
impl Tool for KnowledgeSearch {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_knowledge_base".into(),
            description: format!(
                "Search the knowledge base of indexed public ASU pages. Use when you need facts \
                 you do not already have evidence for. Indexed categories: {}. Leave categories \
                 out to search all of them.",
                self.categories.join(", ")
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to look for." },
                    "categories": {
                        "type": "array",
                        "items": { "type": "string", "enum": self.categories },
                        "description": "Optional categories to restrict to. Omit to search all."
                    }
                },
                "required": ["query"]
            }),
            risk: RiskClass::ReadPublic,
            sequential: false,
            timeout_secs: None,
        }
    }

    async fn call(&self, ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        let args: SearchArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        if args.query.trim().is_empty() {
            return Err(ToolError::InvalidArguments("query is empty".into()));
        }
        // A category the index does not hold matches nothing. Answering with no results reads
        // as an absent fact rather than a wrong filter.
        let unknown = self.unknown(&args.categories);
        if !unknown.is_empty() {
            return Err(ToolError::InvalidArguments(format!(
                "no such category: {}. Indexed categories: {}",
                unknown.join(", "),
                self.categories.join(", ")
            )));
        }
        let q = RetrievalQuery {
            text: args.query,
            categories: args.categories,
            top_k: self.top_k,
        };
        let evidence = self
            .retriever
            .retrieve(ctx, &q)
            .await
            .map_err(|e| ToolError::Failed(e.to_string()))?;
        if evidence.is_empty() {
            return Ok(ToolOutput::text(
                "No indexed source matched. Say you could not find it.",
            ));
        }
        let mut text = String::new();
        for (i, e) in evidence.iter().enumerate() {
            let _ = writeln!(
                text,
                "[{}] {} (fetched {})\n{}\n",
                i + 1,
                e.title,
                e.fetched_at.format("%Y-%m-%d"),
                e.content.trim()
            );
        }
        Ok(ToolOutput {
            content: text,
            data: structured(&evidence),
            evidence,
        })
    }
}
