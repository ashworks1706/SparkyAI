//! Test doubles shared across the suite.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::harness::agent::{Agent, AgentDeps};
use crate::agent::harness::policy::RiskPolicy;
use crate::agent::harness::tool::ToolSet;
use crate::core::traits::model::ModelProvider;
use crate::core::traits::tool::Tool;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::AgentConfig;
use crate::core::types::context::RequestContext;
use crate::core::types::message::ToolCall;
use crate::core::types::model::{FinishReason, ModelError, ModelRequest, ModelResponse, Usage};
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::core::types::trace::{TraceEvent, TraceRecord};

/// Replays canned responses in order.
pub struct Scripted(Mutex<Vec<Result<ModelResponse, ModelError>>>);

impl Scripted {
    pub fn new(items: Vec<Result<ModelResponse, ModelError>>) -> Self {
        let mut reversed = items;
        reversed.reverse();
        Self(Mutex::new(reversed))
    }
}

#[async_trait]
impl ModelProvider for Scripted {
    async fn generate(
        &self,
        _ctx: &RequestContext,
        _req: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        self.0
            .lock()
            .ok()
            .and_then(|mut items| items.pop())
            .unwrap_or_else(|| Err(ModelError::Malformed("script exhausted".into())))
    }
}

pub fn text(content: &str) -> ModelResponse {
    ModelResponse {
        content: content.into(),
        tool_calls: vec![],
        finish_reason: FinishReason::Stop,
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
        },
        model: "test".into(),
    }
}

pub fn calls(items: Vec<(&str, &str, Value)>) -> ModelResponse {
    ModelResponse {
        content: String::new(),
        tool_calls: items
            .into_iter()
            .map(|(id, name, arguments)| ToolCall {
                id: id.into(),
                name: name.into(),
                arguments,
            })
            .collect(),
        finish_reason: FinishReason::ToolCalls,
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
        },
        model: "test".into(),
    }
}

/// Returns its arguments as text.
pub struct Echo(pub RiskClass);

#[async_trait]
impl Tool for Echo {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "echoes".into(),
            parameters: json!({"type": "object"}),
            risk: self.0,
            sequential: false,
        }
    }
    async fn call(&self, _ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput::text(args.to_string()))
    }
}

/// Sleeps past any reasonable tool timeout.
pub struct Slow;

#[async_trait]
impl Tool for Slow {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "slow".into(),
            description: "sleeps".into(),
            parameters: json!({"type": "object"}),
            risk: RiskClass::ReadPublic,
            sequential: false,
        }
    }
    async fn call(&self, _ctx: &RequestContext, _args: Value) -> Result<ToolOutput, ToolError> {
        tokio::time::sleep(Duration::from_secs(5)).await;
        Ok(ToolOutput::text("late"))
    }
}

/// Keeps every trace record so tests can inspect the loop.
#[derive(Default)]
pub struct MemorySink {
    records: Mutex<Vec<TraceRecord>>,
}

impl MemorySink {
    pub fn records(&self) -> Vec<TraceRecord> {
        self.records.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

impl TraceSink for MemorySink {
    fn emit(&self, ctx: &RequestContext, event: TraceEvent) {
        if let Ok(mut records) = self.records.lock() {
            records.push(TraceRecord {
                request_id: ctx.request_id,
                conversation_id: ctx.conversation_id,
                at: chrono::Utc::now(),
                event,
            });
        }
    }
}

/// An agent over a scripted model, with an in-memory trace to inspect.
pub fn agent(model: Scripted, tools: ToolSet, cfg: AgentConfig) -> (Agent, Arc<MemorySink>) {
    let sink = Arc::new(MemorySink::default());
    let deps = AgentDeps {
        model: Arc::new(model),
        tools,
        policy: Arc::new(RiskPolicy::new(Some("Moderator".into()))),
        trace: sink.clone(),
        retriever: None,
        conversations: None,
        memory: None,
    };
    (Agent::new(deps, cfg, "sys"), sink)
}

pub fn ctx() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5))
}

/// Sequential tool: records the order it is called in by sleeping longer for smaller inputs,
/// so parallel execution would reverse the observed order.
pub struct Ordered(pub RiskClass);

#[async_trait]
impl Tool for Ordered {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "ordered".into(),
            description: "stateful".into(),
            parameters: json!({"type": "object"}),
            risk: self.0,
            sequential: true,
        }
    }
    async fn call(&self, _ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        let n = args["n"].as_u64().unwrap_or(0);
        tokio::time::sleep(Duration::from_millis(40 * (4 - n))).await;
        Ok(ToolOutput::text(n.to_string()))
    }
}
