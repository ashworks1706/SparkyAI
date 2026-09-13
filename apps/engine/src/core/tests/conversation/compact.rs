//! Compaction: what it replaces, what it keeps, and what happens when it fails.

use std::sync::Arc;
use std::time::Duration;

use crate::core::tests::support::{Row, Scripted, history_of, push_row, text};
use crate::core::traits::conversation::ConversationStore;
use crate::core::traits::conversation::compaction::Compactor;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Stored;
use crate::core::types::conversation::message::{Message, Role, ToolCall};
use crate::core::types::model::ModelError;
use crate::core::types::store::StoreError;
use crate::runtime::harness::agent::task::{Task, TaskConfig};
use crate::runtime::harness::compact::{ChatCompactor, transcript};

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
                name: "search_library_hours".into(),
                arguments: serde_json::json!({"query": "hayden hours"}),
            }],
        ),
        Message::tool_result("1", "search_library_hours", "2am on weekdays"),
        Message::assistant("Hayden closes at 2am on weekdays."),
    ];
    let out = transcript(&turns);
    assert!(out.contains("User: when does hayden close"), "{out}");
    assert!(out.contains("Sparky called search_library_hours"), "{out}");
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
    // The summary has the summary role and no tool calls.
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

/// A conversation store preloaded with history, keeping positions and summary coverage.
#[derive(Default)]
struct Loaded {
    rows: std::sync::Mutex<Vec<Row>>,
}

impl Loaded {
    fn seed(&self, messages: impl IntoIterator<Item = Message>) {
        if let Ok(mut rows) = self.rows.lock() {
            for m in messages {
                push_row(&mut rows, m, None);
            }
        }
    }

    fn messages(&self) -> Vec<Message> {
        self.rows
            .lock()
            .map(|rows| rows.iter().map(|r| r.message.clone()).collect())
            .unwrap_or_default()
    }
}

#[async_trait::async_trait]
impl ConversationStore for Loaded {
    async fn ensure(&self, _ctx: &RequestContext, _channel_id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    async fn owns(&self, _ctx: &RequestContext) -> Result<bool, StoreError> {
        Ok(true)
    }

    async fn latest(
        &self,
        _ctx: &RequestContext,
        _channel_id: &str,
    ) -> Result<Option<uuid::Uuid>, StoreError> {
        Ok(None)
    }

    async fn end(&self, _ctx: &RequestContext, _channel_id: &str) -> Result<u64, StoreError> {
        Ok(0)
    }

    async fn load(&self, _ctx: &RequestContext, limit: usize) -> Result<Vec<Stored>, StoreError> {
        Ok(self
            .rows
            .lock()
            .map(|rows| history_of(&rows, limit))
            .unwrap_or_default())
    }

    async fn append(&self, _ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError> {
        if let Ok(mut rows) = self.rows.lock() {
            for turn in turns {
                push_row(&mut rows, turn.clone(), None);
            }
        }
        Ok(())
    }

    async fn append_summary(
        &self,
        _ctx: &RequestContext,
        summary: &Message,
        covers: i64,
    ) -> Result<(), StoreError> {
        if let Ok(mut rows) = self.rows.lock() {
            push_row(&mut rows, summary.clone(), Some(covers));
        }
        Ok(())
    }
}

#[tokio::test]
async fn history_over_budget_is_replaced_by_one_turn_that_is_kept() {
    use crate::core::tests::support::MemorySink;
    use crate::core::types::agent::AgentConfig;
    use crate::core::types::agent::assemble::Budget;
    use crate::runtime::harness::agent::{Agent, AgentDeps};
    use crate::runtime::harness::safety::policy::RiskPolicy;
    use crate::runtime::harness::tools::ToolSet;

    let store = Arc::new(Loaded::default());
    store.seed((0..40).flat_map(|i| {
        [
            Message::user(format!("question {i} about ASU library hours and events")),
            Message::assistant(format!("answer {i} with enough text to cost tokens")),
        ]
    }));
    let before = store.messages().len();

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
        guardrail: None,
        profile: None,
        profile_graph: None,
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

    let after = store.messages();
    let summaries: Vec<&Message> = after.iter().filter(|m| m.role == Role::Summary).collect();
    assert_eq!(summaries.len(), 1, "one compacted turn was stored");
    assert!(summaries[0].content.contains("Forty exchanges"));
    // The turns it replaced are still there. The summary stands in for them at assembly time.
    assert!(after.len() > before, "nothing was deleted");
}

#[tokio::test]
async fn a_failed_compaction_leaves_the_run_working() {
    use crate::core::tests::support::MemorySink;
    use crate::core::types::agent::AgentConfig;
    use crate::core::types::agent::assemble::Budget;
    use crate::runtime::harness::agent::{Agent, AgentDeps};
    use crate::runtime::harness::safety::policy::RiskPolicy;
    use crate::runtime::harness::tools::ToolSet;

    let store = Arc::new(Loaded::default());
    store.seed(
        (0..40).map(|i| Message::user(format!("question {i} padded out to spend the budget"))),
    );
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
        guardrail: None,
        profile: None,
        profile_graph: None,
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
    let stored = store.messages();
    assert!(
        !stored.iter().any(|m| m.role == Role::Summary),
        "no summary is stored when compaction fails"
    );
}

#[tokio::test]
async fn the_turns_a_compaction_keeps_are_still_there_on_the_next_request() {
    use crate::core::tests::support::MemorySink;
    use crate::core::types::agent::AgentConfig;
    use crate::core::types::agent::assemble::Budget;
    use crate::runtime::harness::agent::{Agent, AgentDeps};
    use crate::runtime::harness::safety::policy::RiskPolicy;
    use crate::runtime::harness::tools::ToolSet;

    let pad = "x".repeat(80);
    let store = Arc::new(Loaded::default());
    store.seed([
        Message::user(format!("first question {pad}")),
        Message::assistant(format!("first answer {pad}")),
        Message::user(format!("second question {pad}")),
        Message::assistant(format!("second answer {pad}")),
        Message::user(format!("KEPT-QUESTION {pad}")),
        Message::assistant(format!("KEPT-ANSWER {pad}")),
    ]);
    let summarizer = Scripted::new(vec![
        Ok(text("The user asked two questions.")),
        Ok(text("The user asked four questions.")),
    ]);
    let transcripts = summarizer.sent();
    let chat = Scripted::new(vec![
        Ok(text(&format!("fourth answer {pad}"))),
        Ok(text("fifth answer")),
    ]);
    let prompts = chat.sent();
    let deps = AgentDeps {
        model: Arc::new(chat),
        tools: ToolSet::new(),
        policy: Arc::new(RiskPolicy::default()),
        trace: Arc::new(MemorySink::new()),
        retriever: None,
        conversations: Some(store.clone()),
        memory: None,
        confirmations: None,
        compactor: Some(Arc::new(ChatCompactor::new(Task::new(
            Arc::new(summarizer),
            "compaction",
            "compact this",
            TaskConfig::default(),
        )))),
        guardrail: None,
        profile: None,
        profile_graph: None,
    };
    let cfg = AgentConfig {
        budget: Budget {
            history: 60,
            ..Budget::default()
        },
        ..AgentConfig::default()
    };
    let agent = Agent::new(deps, cfg, "sys");

    assert!(agent.run(&ctx(), "turn four").await.is_ok());
    assert!(agent.run(&ctx(), "turn five").await.is_ok());

    let transcripts: Vec<String> = transcripts
        .lock()
        .map(|r| {
            r.iter()
                .map(|q| q.messages.iter().map(|m| m.content.clone()).collect())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(transcripts.len(), 2, "each request compacted once");
    assert!(
        !transcripts[0].contains("KEPT-QUESTION"),
        "the first compaction keeps the newest turns"
    );
    assert!(
        transcripts[1].contains("KEPT-QUESTION")
            && transcripts[1].contains("The user asked two questions."),
        "the second compaction reads what the first kept and the first summary: {}",
        transcripts[1]
    );
    let first_prompt: String = prompts
        .lock()
        .map(|r| r[0].messages.iter().map(|m| m.content.clone()).collect())
        .unwrap_or_default();
    assert!(
        first_prompt.contains("KEPT-ANSWER"),
        "the kept turns reach the prompt"
    );
}
