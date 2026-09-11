//! The chat agent. Turns the history that no longer fits into one turn that does.

use std::fmt::Write as _;

use async_trait::async_trait;

use crate::agent::harness::task::Task;
use crate::core::traits::compaction::Compactor;
use crate::core::types::context::RequestContext;
use crate::core::types::message::{Message, Role};
use crate::core::types::model::ModelError;

/// Default instructions for the chat agent.
pub const INSTRUCTIONS: &str = "You compact a conversation. Rewrite the turns below as one \
short paragraph a later reader can act on. Keep what the user asked for, what was decided, \
every identifier, date and number, and anything still unresolved. Drop pleasantries and \
restatements. Write plain prose, no headings, no bullet points. Do not answer the \
conversation, only record it.";

/// Compacts turns with a prompted sub-agent.
pub struct ChatCompactor {
    task: Task,
}

impl ChatCompactor {
    /// Builds the compactor over a task holding the chat agent instructions.
    pub fn new(task: Task) -> Self {
        Self { task }
    }
}

/// Renders turns as the transcript the chat agent reads.
pub fn transcript(turns: &[Message]) -> String {
    let mut out = String::new();
    for m in turns {
        let who = match m.role {
            Role::User => "User",
            Role::Assistant => "Sparky",
            Role::Tool => "Tool result",
            Role::System => "Instructions",
            Role::Summary => "Earlier summary",
        };
        let text = m.content.trim();
        if !text.is_empty() {
            let _ = writeln!(out, "{who}: {text}");
        }
        for call in &m.tool_calls {
            let _ = writeln!(out, "Sparky called {} with {}", call.name, call.arguments);
        }
    }
    out
}

#[async_trait]
impl Compactor for ChatCompactor {
    async fn compact(
        &self,
        ctx: &RequestContext,
        turns: &[Message],
    ) -> Result<Message, ModelError> {
        let text = transcript(turns);
        if text.trim().is_empty() {
            return Err(ModelError::Malformed("nothing to compact".into()));
        }
        Ok(Message::summary(self.task.run(ctx, &text).await?))
    }
}
