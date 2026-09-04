//! Test doubles shared across the suite.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::core::traits::model::ModelProvider;
use crate::core::traits::tool::Tool;
use crate::core::types::agent::AgentConfig;
use crate::core::types::context::RequestContext;
use crate::core::types::harness::{Agent, AgentDeps, MemorySink, RiskPolicy, ToolSet};
use crate::core::types::message::ToolCall;
use crate::core::types::model::{FinishReason, ModelError, ModelRequest, ModelResponse, Usage};
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};

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
        }
    }
    async fn call(&self, _ctx: &RequestContext, _args: Value) -> Result<ToolOutput, ToolError> {
        tokio::time::sleep(Duration::from_secs(5)).await;
        Ok(ToolOutput::text("late"))
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
