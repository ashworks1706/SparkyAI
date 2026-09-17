//! A prompted sub-agent: one prompt, one model call, no tools.

use std::sync::Arc;
use std::time::Duration;

use tracing::Instrument;
use tracing::field::Empty;

use crate::core::traits::model::ModelProvider;
use crate::core::types::agent::AgentConfig;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::message::Message;
use crate::core::types::model::{ModelError, ModelRequest};
use crate::runtime::harness::safety::redact::{json, truncate};

/// What a task sends with every call.
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// Completion budget.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Wall-clock budget for the call.
    pub timeout: Duration,
    /// gen_ai.provider.name on the span.
    pub provider_name: Arc<str>,
    /// Model name as requested, for gen_ai.request.model.
    pub model_name: Arc<str>,
    /// Longest value recorded on the span.
    pub max_span_value_chars: usize,
}

impl Default for TaskConfig {
    fn default() -> Self {
        let agent = AgentConfig::default();
        Self {
            max_tokens: 512,
            temperature: 0.0,
            timeout: Duration::from_secs(30),
            provider_name: agent.provider_name,
            model_name: agent.model_name,
            max_span_value_chars: agent.max_span_value_chars,
        }
    }
}

/// One prompted sub-agent, named for the trace.
pub struct Task {
    model: Arc<dyn ModelProvider>,
    name: &'static str,
    instructions: Arc<str>,
    cfg: TaskConfig,
}

impl Task {
    /// Builds a task that answers with the given instructions.
    pub fn new(
        model: Arc<dyn ModelProvider>,
        name: &'static str,
        instructions: impl Into<Arc<str>>,
        cfg: TaskConfig,
    ) -> Self {
        Self {
            model,
            name,
            instructions: instructions.into(),
            cfg,
        }
    }

    /// Runs the instructions over input and returns the text.
    pub async fn run(&self, ctx: &RequestContext, input: &str) -> Result<String, ModelError> {
        let request = ModelRequest {
            messages: vec![
                Message::system(self.instructions.as_ref()),
                Message::user(input),
            ],
            tools: Vec::new(),
            max_tokens: self.cfg.max_tokens,
            temperature: self.cfg.temperature,
            thinking: false,
        };
        let limit = self.cfg.max_span_value_chars;
        let prompt = truncate(&json(&request.messages), limit);
        let span = tracing::info_span!(
            "task",
            "gen_ai.operation.name" = "chat",
            "gen_ai.provider.name" = %self.cfg.provider_name,
            "gen_ai.request.model" = %self.cfg.model_name,
            "gen_ai.response.model" = Empty,
            "gen_ai.request.max_tokens" = request.max_tokens,
            "gen_ai.request.temperature" = f64::from(request.temperature),
            "gen_ai.usage.input_tokens" = Empty,
            "gen_ai.usage.output_tokens" = Empty,
            "gen_ai.input.messages" = %prompt,
            "gen_ai.output.messages" = Empty,
            // OpenInference, read by the Phoenix trace UI.
            "openinference.span.kind" = "LLM",
            "input.value" = %prompt,
            "input.mime_type" = "application/json",
            "output.value" = Empty,
            "output.mime_type" = "application/json",
            "llm.model_name" = Empty,
            "llm.token_count.prompt" = Empty,
            "llm.token_count.completion" = Empty,
            "session.id" = %ctx.conversation_id,
            "user.id" = %ctx.user_id,
            "sparky.span" = "task",
            "sparky.task" = self.name,
            "sparky.tools" = "[]",
            "sparky.step" = 0,
            "sparky.attempt" = 0,
        );
        let budget = self.cfg.timeout.min(ctx.remaining());
        let call = self.model.generate(ctx, request).instrument(span.clone());
        let response = tokio::time::timeout(budget, call)
            .await
            .map_err(|_| ModelError::Transport(format!("{} timed out", self.name)))??;
        super::call::record_reply(&span, &response, limit);
        let text = response.content.trim().to_owned();
        if text.is_empty() {
            return Err(ModelError::Malformed(format!(
                "{} answered with nothing",
                self.name
            )));
        }
        Ok(text)
    }
}
