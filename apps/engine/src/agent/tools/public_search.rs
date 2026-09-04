//! `ReadPublic`: search indexed ASU sources through `Retriever`.

use std::fmt::Write;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::core::traits::retrieval::Retriever;
use crate::core::traits::tool::Tool;
use crate::core::types::adapters::{PublicSearch, SearchArgs};
use crate::core::types::context::RequestContext;
use crate::core::types::retrieval::RetrievalQuery;
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};

impl PublicSearch {
    /// Searches with `retriever`, returning at most `top_k` chunks.
    pub fn new(retriever: Arc<dyn Retriever>, top_k: usize) -> Self {
        Self { retriever, top_k }
    }
}

#[async_trait]
impl Tool for PublicSearch {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_asu".into(),
            description:
                "Search indexed public ASU sources (library hours, events, clubs, courses, \
                          scholarships, news, shuttles, jobs, sports). Use when you need facts you \
                          do not already have evidence for."
                    .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to look for." },
                    "categories": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional source categories to restrict to."
                    }
                },
                "required": ["query"]
            }),
            risk: RiskClass::ReadPublic,
        }
    }

    async fn call(&self, ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        let args: SearchArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        if args.query.trim().is_empty() {
            return Err(ToolError::InvalidArguments("query is empty".into()));
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
            data: serde_json::to_value(&evidence).ok(),
        })
    }
}
