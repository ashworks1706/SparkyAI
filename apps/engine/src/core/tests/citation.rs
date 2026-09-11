//! What an answer may cite, and what tool output leaves behind in a trace.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::agent::harness::agent::redact_text;
use crate::core::tests::support::{agent, calls, text};
use crate::core::traits::retrieval::Retriever;
use crate::core::traits::tool::Tool;
use crate::core::types::agent::AgentConfig;
use crate::core::types::assemble::Budget;
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::retrieval::{RetrievalError, RetrievalQuery};
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};

fn ctx() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5))
}

fn evidence(title: &str, chars: usize) -> Evidence {
    Evidence {
        source_id: Uuid::new_v4(),
        chunk_id: Uuid::new_v4(),
        title: title.into(),
        content: "x".repeat(chars),
        url: None,
        fetched_at: Utc::now(),
        score: 1.0,
    }
}

/// Hands back more evidence than any budget holds.
struct Flood(Vec<Evidence>);

#[async_trait]
impl Retriever for Flood {
    async fn retrieve(
        &self,
        _ctx: &RequestContext,
        _query: &RetrievalQuery,
    ) -> Result<Vec<Evidence>, RetrievalError> {
        Ok(self.0.clone())
    }
}

/// A tool that finds one indexed chunk.
struct Finder(Evidence);

#[async_trait]
impl Tool for Finder {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "finder".into(),
            description: "finds".into(),
            parameters: json!({"type": "object"}),
            risk: RiskClass::ReadPublic,
            sequential: false,
            timeout_secs: None,
        }
    }
    async fn call(&self, _ctx: &RequestContext, _args: Value) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput {
            content: "found one".into(),
            data: None,
            evidence: vec![self.0.clone()],
        })
    }
}

#[tokio::test]
async fn a_chunk_that_did_not_fit_the_prompt_is_not_cited() {
    use crate::agent::harness::agent::{Agent, AgentDeps};
    use crate::agent::harness::policy::RiskPolicy;
    use crate::agent::harness::tool::ToolSet;
    use crate::core::tests::support::{MemorySink, Scripted};

    let flood: Vec<Evidence> = (0..20)
        .map(|i| evidence(&format!("doc {i}"), 400))
        .collect();
    let deps = AgentDeps {
        model: Arc::new(Scripted::new(vec![Ok(text("answered"))])),
        tools: ToolSet::new(),
        policy: Arc::new(RiskPolicy::default()),
        trace: Arc::new(MemorySink::new()),
        retriever: Some(Arc::new(Flood(flood))),
        conversations: None,
        memory: None,
        confirmations: None,
        compactor: None,
        guardrail: None,
        profile: None,
        profile_graph: None,
    };
    let cfg = AgentConfig {
        retrieval_top_k: 20,
        budget: Budget {
            evidence: 500,
            ..Budget::default()
        },
        ..AgentConfig::default()
    };
    let Ok(answer) = Agent::new(deps, cfg, "sys").run(&ctx(), "q").await else {
        unreachable!("the run answered")
    };
    assert!(!answer.evidence.is_empty(), "something was cited");
    assert!(
        answer.evidence.len() < 20,
        "citing all twenty would attribute the answer to chunks the model never saw, got {}",
        answer.evidence.len()
    );
}

#[tokio::test]
async fn a_chunk_a_tool_found_is_cited() {
    use crate::agent::harness::agent::{Agent, AgentDeps};
    use crate::agent::harness::policy::RiskPolicy;
    use crate::agent::harness::tool::ToolSet;
    use crate::core::tests::support::{MemorySink, Scripted};

    let found = evidence("from the tool", 10);
    let wanted = found.chunk_id;
    let deps = AgentDeps {
        model: Arc::new(Scripted::new(vec![
            Ok(calls(vec![("1", "finder", json!({}))])),
            Ok(text("answered")),
        ])),
        tools: ToolSet::new().with(Arc::new(Finder(found))),
        policy: Arc::new(RiskPolicy::default()),
        trace: Arc::new(MemorySink::new()),
        retriever: None,
        conversations: None,
        memory: None,
        confirmations: None,
        compactor: None,
        guardrail: None,
        profile: None,
        profile_graph: None,
    };
    let Ok(answer) = Agent::new(deps, AgentConfig::default(), "sys")
        .run(&ctx(), "q")
        .await
    else {
        unreachable!("the run answered")
    };
    assert!(
        answer.evidence.iter().any(|e| e.chunk_id == wanted),
        "what a tool found is cited too"
    );
}

#[tokio::test]
async fn nothing_retrieved_cites_nothing() {
    let (a, _sink) = agent(
        crate::core::tests::support::Scripted::new(vec![Ok(text("hello"))]),
        crate::agent::harness::tool::ToolSet::new(),
        AgentConfig::default(),
    );
    let Ok(answer) = a.run(&ctx(), "hi").await else {
        unreachable!("the run answered")
    };
    assert!(answer.evidence.is_empty());
}

#[test]
fn a_secret_in_tool_output_is_masked_before_it_reaches_a_trace() {
    let page = "Welcome back\nAuthorization: Bearer abc123\nsession_token = zzz\nordinary line";
    let out = redact_text(page);
    assert!(!out.contains("abc123"), "{out}");
    assert!(!out.contains("zzz"), "{out}");
    assert!(out.contains("Welcome back"), "{out}");
    assert!(out.contains("ordinary line"), "{out}");
    assert!(out.contains("[redacted]"), "{out}");
}

#[test]
fn redaction_leaves_text_that_carries_no_secret_alone() {
    let page = "Hayden Library closes at 2am.\nAsk at the front desk.";
    assert_eq!(redact_text(page), page);
}

#[test]
fn a_key_with_no_value_after_it_is_left_as_it_reads() {
    // A sentence mentioning the word token is not a secret.
    let page = "You will need a token to continue";
    assert_eq!(redact_text(page), page);
}
