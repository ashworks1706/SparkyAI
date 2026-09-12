//! Model doubles: a scripted provider and the responses it replays.

use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use crate::core::traits::model::ModelProvider;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::message::ToolCall;
use crate::core::types::model::{FinishReason, ModelError, ModelRequest, ModelResponse, Usage};

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
        reasoning: String::new(),
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
        reasoning: String::new(),
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
