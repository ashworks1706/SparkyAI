//! Keeping the turns, recording the outcome, and building the answer.

use super::{Agent, ms};
use crate::agent::harness::agent::run::{Run, cited};
use crate::core::types::agent::context::RequestContext;
use crate::core::types::agent::{AgentError, Answer};
use crate::core::types::conversation::Visibility;
use crate::core::types::conversation::message::{Message, Role};
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::safety::policy::ConfirmationRequest;
use crate::core::types::trace::{RunStatus, TraceEvent};

impl Agent {
    /// Keeps the turns, records the outcome, and builds the answer. Every exit from the loop
    /// goes through here.
    pub(super) async fn conclude(
        &self,
        run: &Run<'_>,
        status: RunStatus,
        text: String,
        evidence: Vec<Evidence>,
        confirmation: Option<ConfirmationRequest>,
    ) -> Result<Answer, AgentError> {
        self.persist(run.ctx, &run.new_turns).await?;
        self.record_profile(run);
        let evidence = cited(evidence, run);
        let text = if text.trim().is_empty() {
            status.explain().unwrap_or_default().to_owned()
        } else {
            text
        };
        Ok(self.finish(run, status, text, evidence, confirmation))
    }
    /// Hands the turn to the profile writer and returns. The spawned task carries its own
    /// context and deadline, so the answer never waits on it.
    fn record_profile(&self, run: &Run<'_>) {
        let Some(writer) = self.deps.profile.clone() else {
            return;
        };
        let turn = run
            .new_turns
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join("\n");
        if turn.trim().is_empty() {
            return;
        }
        let (tenant, user) = (run.ctx.tenant_id.clone(), run.ctx.user_id.clone());
        tokio::spawn(async move { writer.record(tenant, user, turn).await });
    }
    async fn persist(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), AgentError> {
        if let Some(store) = &self.deps.conversations {
            store
                .append(ctx, turns)
                .await
                .map_err(|error| AgentError::Store(format!("persist: {error}")))?;
        }
        Ok(())
    }
    fn finish(
        &self,
        run: &Run<'_>,
        status: RunStatus,
        text: String,
        evidence: Vec<Evidence>,
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
            evidence,
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
