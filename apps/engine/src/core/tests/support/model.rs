//! Model doubles: a scripted provider and the responses it replays.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::core::traits::model::ModelProvider;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::message::ToolCall;
use crate::core::types::model::{
    FinishReason, ModelDelta, ModelError, ModelRequest, ModelResponse, Usage,
};

/// Replays canned responses in order and keeps every request it was sent.
pub struct Scripted {
    items: Mutex<Vec<Result<ModelResponse, ModelError>>>,
    sent: Arc<Mutex<Vec<ModelRequest>>>,
    streams: bool,
}

impl Scripted {
    pub fn new(items: Vec<Result<ModelResponse, ModelError>>) -> Self {
        let mut reversed = items;
        reversed.reverse();
        Self {
            items: Mutex::new(reversed),
            sent: Arc::default(),
            streams: false,
        }
    }

    /// The same script, streamed: each response is sent word by word, reasoning first.
    pub fn streaming(mut self) -> Self {
        self.streams = true;
        self
    }

    /// The requests this model is sent, readable after it moves into an agent.
    pub fn sent(&self) -> Arc<Mutex<Vec<ModelRequest>>> {
        Arc::clone(&self.sent)
    }
}

#[async_trait]
impl ModelProvider for Scripted {
    async fn generate(
        &self,
        _ctx: &RequestContext,
        req: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        if let Ok(mut sent) = self.sent.lock() {
            sent.push(req);
        }
        self.next()
    }

    async fn stream(
        &self,
        ctx: &RequestContext,
        req: ModelRequest,
        deltas: UnboundedSender<ModelDelta>,
    ) -> Result<ModelResponse, ModelError> {
        let response = self.generate(ctx, req).await;
        if self.streams
            && let Ok(done) = &response
        {
            for piece in done.reasoning.split_inclusive(' ') {
                let _ = deltas.send(ModelDelta::Reasoning(piece.to_owned()));
            }
            for piece in done.content.split_inclusive(' ') {
                let _ = deltas.send(ModelDelta::Text(piece.to_owned()));
            }
        }
        response
    }
}

impl Scripted {
    fn next(&self) -> Result<ModelResponse, ModelError> {
        self.items
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

/// A completion that reasoned and left no answer.
pub fn only_thought(reasoning: &str) -> ModelResponse {
    ModelResponse {
        reasoning: reasoning.into(),
        finish_reason: FinishReason::Unknown,
        ..text("")
    }
}
