//! Compaction: what it replaces, what it keeps, and what happens when it fails.

use std::sync::Arc;
use std::time::Duration;

use crate::agent::harness::compact::{ChatCompactor, transcript};
use crate::agent::harness::task::{Task, TaskConfig};
use crate::core::tests::support::{Scripted, text};
use crate::core::traits::compaction::Compactor;
use crate::core::types::context::RequestContext;
use crate::core::types::message::{Message, Role, ToolCall};
use crate::core::types::model::ModelError;

fn ctx() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5))
}

fn compactor(
    replies: Vec<Result<crate::core::types::model::ModelResponse, ModelError>>,
) -> ChatCompactor {
    ChatCompactor::new(Task::new(
        Arc::new(Scripted::new(replies)),
        "compaction",
        "compact this",
        TaskConfig::default(),
    ))
}

#[test]
fn the_transcript_names_who_said_what_and_keeps_tool_calls() {
    let turns = vec![
        Message::user("when does hayden close"),
        Message::assistant_tool_calls(
            "",
            vec![ToolCall {
                id: "1".into(),
                name: "search_knowledge_base".into(),
                arguments: serde_json::json!({"query": "hayden hours"}),
            }],
        ),
        Message::tool_result("1", "search_knowledge_base", "2am on weekdays"),
        Message::assistant("Hayden closes at 2am on weekdays."),
    ];
    let out = transcript(&turns);
    assert!(out.contains("User: when does hayden close"), "{out}");
    assert!(out.contains("Sparky called search_knowledge_base"), "{out}");
    assert!(out.contains("Tool result: 2am on weekdays"), "{out}");
    assert!(out.contains("Sparky: Hayden closes"), "{out}");
}

#[tokio::test]
async fn a_compacted_turn_carries_its_own_role() {
    let c = compactor(vec![Ok(text(
        "The student asked about Hayden hours. It closes at 2am.",
    ))]);
    let turns = vec![Message::user("when does hayden close")];
    let Ok(summary) = c.compact(&ctx(), &turns).await else {
        unreachable!("the model answered")
    };
    // A summary that looked like an assistant turn would be indistinguishable from what
    // Sparky actually said when the conversation is replayed.
    assert_eq!(summary.role, Role::Summary);
    assert!(summary.content.contains("Hayden"));
    assert!(summary.tool_calls.is_empty());
}

#[tokio::test]
async fn nothing_to_compact_is_an_error_rather_than_an_empty_summary() {
    let c = compactor(vec![Ok(text("unused"))]);
    let blank = vec![Message::assistant("")];
    assert!(c.compact(&ctx(), &blank).await.is_err());
}

#[tokio::test]
async fn a_model_that_answers_with_nothing_is_an_error() {
    let c = compactor(vec![Ok(text("   "))]);
    let turns = vec![Message::user("something")];
    assert!(c.compact(&ctx(), &turns).await.is_err());
}

/// A conversation store preloaded with history, recording what the loop appends.
#[derive(Default)]
struct Loaded {
    turns: std::sync::Mutex<Vec<Message>>,
}

#[async_trait::async_trait]
impl crate::core::traits::conversation::ConversationStore for Loaded {
    async fn ensure(
        &self,
        _ctx: &RequestContext,
        _channel_id: &str,
    ) -> Result<(), crate::core::types::store::StoreError> {
        Ok(())
    }

    async fn load(
        &self,
        _ctx: &RequestContext,
        _limit: usize,
    ) -> Result<Vec<Message>, crate::core::types::store::StoreError> {
        Ok(self.turns.lock().map(|t| t.clone()).unwrap_or_default())
    }

    async fn append(
        &self,
        _ctx: &RequestContext,
        turns: &[Message],
    ) -> Result<(), crate::core::types::store::StoreError> {
        if let Ok(mut kept) = self.turns.lock() {
            kept.extend_from_slice(turns);
        }
        Ok(())
    }
}

#[tokio::test]
async fn history_over_budget_is_replaced_by_one_turn_that_is_kept() {
    use crate::agent::harness::agent::{Agent, AgentDeps};
    use crate::agent::harness::policy::RiskPolicy;
    use crate::agent::harness::tool::ToolSet;
    use crate::core::tests::support::MemorySink;
    use crate::core::types::agent::AgentConfig;
    use crate::core::types::assemble::Budget;

    let store = Arc::new(Loaded::default());
    if let Ok(mut turns) = store.turns.lock() {
        for i in 0..40 {
            turns.push(Message::user(format!(
                "question {i} about ASU library hours and events"
            )));
            turns.push(Message::assistant(format!(
                "answer {i} with enough text to cost tokens"
            )));
        }
    }
    let before = store.turns.lock().map_or(0, |t| t.len());

    let deps = AgentDeps {
        model: Arc::new(Scripted::new(vec![Ok(text("done"))])),
        tools: ToolSet::new(),
        policy: Arc::new(RiskPolicy::default()),
        trace: Arc::new(MemorySink::new()),
        retriever: None,
        conversations: Some(store.clone()),
        memory: None,
        confirmations: None,
        compactor: Some(Arc::new(compactor(vec![Ok(text(
            "Forty exchanges about hours.",
        ))]))),
    };
    let cfg = AgentConfig {
        budget: Budget {
            history: 200,
            ..Budget::default()
        },
        ..AgentConfig::default()
    };
    let agent = Agent::new(deps, cfg, "sys");
    let Ok(answer) = agent.run(&ctx(), "and now?").await else {
        unreachable!("the run completed")
    };
    assert_eq!(answer.text, "done");

    let Ok(after) = store.turns.lock().map(|t| t.clone()) else {
        unreachable!("the store is readable")
    };
    let summaries: Vec<&Message> = after.iter().filter(|m| m.role == Role::Summary).collect();
    assert_eq!(summaries.len(), 1, "one compacted turn was stored");
    assert!(summaries[0].content.contains("Forty exchanges"));
    // The turns it replaced are still there. The summary stands in for them at assembly time.
    assert!(after.len() > before, "nothing was deleted");
}

#[tokio::test]
async fn a_failed_compaction_leaves_the_run_working() {
    use crate::agent::harness::agent::{Agent, AgentDeps};
    use crate::agent::harness::policy::RiskPolicy;
    use crate::agent::harness::tool::ToolSet;
    use crate::core::tests::support::MemorySink;
    use crate::core::types::agent::AgentConfig;
    use crate::core::types::assemble::Budget;

    let store = Arc::new(Loaded::default());
    if let Ok(mut turns) = store.turns.lock() {
        for i in 0..40 {
            turns.push(Message::user(format!(
                "question {i} padded out to spend the budget"
            )));
        }
    }
    let deps = AgentDeps {
        model: Arc::new(Scripted::new(vec![Ok(text("answered anyway"))])),
        tools: ToolSet::new(),
        policy: Arc::new(RiskPolicy::default()),
        trace: Arc::new(MemorySink::new()),
        retriever: None,
        conversations: Some(store.clone()),
        memory: None,
        confirmations: None,
        // The compactor has no scripted reply, so its model call fails.
        compactor: Some(Arc::new(compactor(Vec::new()))),
    };
    let cfg = AgentConfig {
        budget: Budget {
            history: 200,
            ..Budget::default()
        },
        ..AgentConfig::default()
    };
    let agent = Agent::new(deps, cfg, "sys");
    let Ok(answer) = agent.run(&ctx(), "and now?").await else {
        unreachable!("a failed compaction does not fail the request")
    };
    assert_eq!(answer.text, "answered anyway");
    let stored = store.turns.lock().map(|t| t.clone()).unwrap_or_default();
    assert!(
        !stored.iter().any(|m| m.role == Role::Summary),
        "no summary is stored when compaction fails"
    );
}
