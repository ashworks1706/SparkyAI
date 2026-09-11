//! A prompted sub-agent. One prompt, one model call, no tools, a typed result.
//!
//! The loop is not built on this. The loop has tools and stopping conditions; a task has one
//! instruction and one answer.

use std::sync::Arc;
use std::time::Duration;

use tracing::Instrument;
use tracing::field::Empty;

use crate::core::traits::model::ModelProvider;
use crate::core::types::context::RequestContext;
use crate::core::types::message::Message;
use crate::core::types::model::{ModelError, ModelRequest};

/// What a task sends with every call.
#[derive(Debug, Clone, Copy)]
pub struct TaskConfig {
    /// Completion budget.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Wall-clock budget for the call.
    pub timeout: Duration,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            // A task rewrites or classifies. Sampling wide serves neither.
            temperature: 0.0,
            timeout: Duration::from_secs(30),
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
    ///
    /// # Errors
    /// Returns [`ModelError`] when the call fails or answers with nothing.
    pub async fn run(&self, ctx: &RequestContext, input: &str) -> Result<String, ModelError> {
        let span = tracing::info_span!(
            "task",
            "openinference.span.kind" = "LLM",
            "sparky.task" = self.name,
            "input.value" = %input,
            "output.value" = Empty,
        );
        let request = ModelRequest {
            messages: vec![
                Message::system(self.instructions.as_ref()),
                Message::user(input),
            ],
            tools: Vec::new(),
            max_tokens: self.cfg.max_tokens,
            temperature: self.cfg.temperature,
        };
        let budget = self.cfg.timeout.min(ctx.remaining());
        let call = self.model.generate(ctx, request).instrument(span.clone());
        let response = tokio::time::timeout(budget, call)
            .await
            .map_err(|_| ModelError::Transport(format!("{} timed out", self.name)))??;
        let text = response.content.trim().to_owned();
        if text.is_empty() {
            return Err(ModelError::Malformed(format!(
                "{} answered with nothing",
                self.name
            )));
        }
        span.record("output.value", text.as_str());
        Ok(text)
    }
}
