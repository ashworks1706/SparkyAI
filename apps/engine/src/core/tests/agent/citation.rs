//! What an answer may cite, and what tool output leaves behind in a trace.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::core::tests::support::{agent, calls, text};
use crate::core::traits::knowledge::retrieval::Retriever;
use crate::core::traits::tools::Tool;
use crate::core::types::agent::AgentConfig;
use crate::core::types::agent::assemble::Budget;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::evidence::{Citation, Evidence};
use crate::core::types::knowledge::retrieval::{RetrievalError, RetrievalQuery};
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::harness::safety::redact::redact_text;

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

/// A tool that reads one live page.
struct Finder(Citation);

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
            sources: vec![self.0.clone()],
        })
    }
}

#[tokio::test]
async fn a_chunk_that_did_not_fit_the_prompt_is_not_cited() {
    use crate::core::tests::support::{MemorySink, Scripted};
    use crate::runtime::harness::agent::{Agent, AgentDeps};
    use crate::runtime::harness::safety::policy::RiskPolicy;
    use crate::runtime::harness::tools::ToolSet;

    let flood: Vec<Evidence> = (0..20)
        .map(|i| evidence(&format!("doc {i}"), 400))
        .collect();
    let deps = AgentDeps {
        model: Arc::new(Scripted::new(vec![Ok(text("answered"))])),
        tools: ToolSet::new(),
        policy: Arc::new(RiskPolicy::default()),
        trace: Arc::new(MemorySink::new()),
        retriever: Some(Arc::new(Flood(flood))),
        router: None,
        conversations: None,
        memory: None,
        confirmations: None,
        sandbox: None,
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
async fn a_page_a_tool_read_is_cited() {
    use crate::core::tests::support::{MemorySink, Scripted};
    use crate::runtime::harness::agent::{Agent, AgentDeps};
    use crate::runtime::harness::safety::policy::RiskPolicy;
    use crate::runtime::harness::tools::ToolSet;

    let found = Citation {
        title: "courses".into(),
        url: Some("https://catalog.apps.asu.edu/catalog/classes/classlist".into()),
    };
    let wanted = found.clone();
    let deps = AgentDeps {
        model: Arc::new(Scripted::new(vec![
            Ok(calls(vec![("1", "finder", json!({}))])),
            Ok(text("answered")),
        ])),
        tools: ToolSet::new().with(Arc::new(Finder(found))),
        policy: Arc::new(RiskPolicy::default()),
        trace: Arc::new(MemorySink::new()),
        retriever: None,
        router: None,
        conversations: None,
        memory: None,
        confirmations: None,
        sandbox: None,
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
        answer.evidence.is_empty(),
        "a live page is not retrieval evidence"
    );
    assert_eq!(
        answer.citations(),
        vec![wanted],
        "what a tool read is cited"
    );
}

#[tokio::test]
async fn nothing_retrieved_cites_nothing() {
    let (a, _sink) = agent(
        crate::core::tests::support::Scripted::new(vec![Ok(text("hello"))]),
        crate::runtime::harness::tools::ToolSet::new(),
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

#[test]
fn citations_list_each_source_once_best_first() {
    use crate::core::types::knowledge::evidence::Evidence;

    let chunk = |source: Uuid, title: &str, url: Option<&str>| Evidence {
        source_id: source,
        chunk_id: Uuid::new_v4(),
        title: title.into(),
        content: String::new(),
        url: url.map(str::to_owned),
        fetched_at: Utc::now(),
        score: 1.0,
    };
    let hours = Uuid::new_v4();
    let events = Uuid::new_v4();
    let note = Uuid::new_v4();
    let evidence = vec![
        chunk(hours, "library_hours", Some("https://lib.asu.edu/hours")),
        chunk(events, "events", Some("https://asu.edu/events")),
        chunk(hours, "library_hours", Some("https://lib.asu.edu/hours")),
        chunk(note, "note", None),
        chunk(hours, "library_hours", Some("https://lib.asu.edu/hours")),
        chunk(note, "note", None),
    ];

    let lines = Evidence::citations(&evidence);

    assert_eq!(lines.len(), 3, "{lines:?}");
    assert_eq!(lines[0].title, "library_hours");
    assert_eq!(lines[0].url.as_deref(), Some("https://lib.asu.edu/hours"));
    assert_eq!(lines[1].title, "events");
    assert_eq!(lines[2].title, "note");
    assert_eq!(lines[2].url, None, "a source with no page of its own");
}
