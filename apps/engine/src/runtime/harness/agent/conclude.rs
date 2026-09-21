//! Keeping the turns, recording the outcome, and building the answer.

use super::{Agent, ms};
use crate::core::types::agent::context::RequestContext;
use crate::core::types::agent::{AgentError, Answer};
use crate::core::types::conversation::Visibility;
use crate::core::types::conversation::message::{Message, Role};
use crate::core::types::safety::policy::ConfirmationRequest;
use crate::core::types::trace::{RunStatus, TraceEvent};
use crate::runtime::harness::agent::run::Run;

/// Whether a turn of this request belongs in the stored conversation.
///
/// A tool call and its result answer the question being asked now. Carrying them into later turns
/// replays a page the student never asked about again and spends the history budget on it, so the
/// stored conversation is what was said: the question and the answer.
pub(super) fn said(message: &Message) -> bool {
    match message.role {
        Role::User => true,
        Role::Assistant => message.tool_calls.is_empty() && !message.content.trim().is_empty(),
        Role::System | Role::Tool | Role::Summary => false,
    }
}

impl Agent {
    /// Keeps the turns, records the outcome, builds the answer. Every loop exit goes through here.
    pub(super) async fn conclude(
        &self,
        run: &Run<'_>,
        status: RunStatus,
        text: String,
        confirmation: Option<ConfirmationRequest>,
    ) -> Result<Answer, AgentError> {
        let said: Vec<Message> = run.new_turns.iter().filter(|m| said(m)).cloned().collect();
        self.persist(run.ctx, &said).await?;
        self.record_profile(run);
        let text = if text.trim().is_empty() {
            status.explain().unwrap_or_default().to_owned()
        } else {
            text
        };
        Ok(self.finish(run, status, text, confirmation))
    }

    /// Hands the user turns to the profile writer on a detached task.
    fn record_profile(&self, run: &Run<'_>) {
        let Some(writer) = self.deps.profile.clone() else {
            return;
        };
        let turn = run
            .new_turns
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if turn.trim().is_empty() {
            return;
        }
        let (tenant, user) = (run.ctx.tenant_id.clone(), run.ctx.user_id.clone());
        tokio::spawn(async move { writer.record(tenant, user, turn).await });
    }

    /// Appends the turns of this request to the conversation store, when one is configured.
    async fn persist(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), AgentError> {
        if let Some(store) = &self.deps.conversations {
            store
                .append(ctx, turns)
                .await
                .map_err(|error| AgentError::Store(format!("persist: {error}")))?;
        }
        Ok(())
    }

    /// Emits the completion event and builds the answer.
    fn finish(
        &self,
        run: &Run<'_>,
        status: RunStatus,
        text: String,
        confirmation: Option<ConfirmationRequest>,
    ) -> Answer {
        let cost_usd = self.cost(run.usage);
        self.deps.trace.emit(
            run.ctx,
            TraceEvent::Completed {
                status: status.clone(),
                steps: run.steps,
                usage: run.usage,
                cost_usd,
                duration_ms: ms(run.started),
            },
        );
        Answer {
            text,
            sources: run.tool_sources.clone(),
            confirmation,
            status,
            steps: run.steps,
            tool_runs: run.tool_runs.clone(),
            // A public answer never names a memory, whatever the prompt carried.
            memories: if run.ctx.visibility == Visibility::Public {
                Vec::new()
            } else {
                run.memories_in_prompt.clone()
            },
            usage: run.usage,
            cost_usd,
        }
    }
}
