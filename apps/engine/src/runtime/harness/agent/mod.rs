//! The agent loop: model call, policy, tool execution, repeated until final answer, error,
//! cancel, deadline, or step limit.

pub mod call;
mod conclude;
mod execute;
mod inputs;
pub mod prompt;
mod run;
mod step;
pub mod task;

use std::sync::Arc;
use std::time::Instant;

use tracing::Instrument;
use tracing::field::Empty;

use crate::core::traits::conversation::ConversationStore;
use crate::core::traits::conversation::compaction::Compactor;
use crate::core::traits::knowledge::retrieval::Retriever;
use crate::core::traits::memory::MemoryStore;
use crate::core::traits::memory::profile::ProfileGraph;
use crate::core::traits::model::ModelProvider;
use crate::core::traits::safety::confirmation::ConfirmationStore;
use crate::core::traits::safety::guardrail::Guardrail;
use crate::core::traits::safety::policy::Policy;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::agent::{AgentConfig, AgentError, Answer};
use crate::core::types::conversation::message::{Message, ToolCall};
use crate::core::types::model::{ModelError, Usage};
use crate::core::types::safety::policy::{ConfirmationRequest, PendingAction};
use crate::core::types::tools::ToolRun;
use crate::core::types::trace::{RunStatus, TraceEvent};
use crate::runtime::harness::agent::run::{Inputs, Run};
use crate::runtime::harness::memory::profile::ProfileWriter;
use crate::runtime::harness::tools::ToolSet;

pub use crate::runtime::harness::agent::prompt::PromptText;
use crate::runtime::harness::safety::redact::truncate;

/// The dependencies the loop drives. Every one is a trait with a test double.
pub struct AgentDeps {
    /// The chat model.
    pub model: Arc<dyn ModelProvider>,
    /// Tools the model may call.
    pub tools: ToolSet,
    /// Gates every tool call.
    pub policy: Arc<dyn Policy>,
    /// Receives every event.
    pub trace: Arc<dyn TraceSink>,
    /// Evidence, when configured.
    pub retriever: Option<Arc<dyn Retriever>>,
    /// Conversation history, when configured.
    pub conversations: Option<Arc<dyn ConversationStore>>,
    /// Cross-conversation memory, when configured.
    pub memory: Option<Arc<dyn MemoryStore>>,
    /// Where actions wait for approval by the caller, when configured.
    pub confirmations: Option<Arc<dyn ConfirmationStore>>,
    /// Compacts history that no longer fits, when configured.
    pub compactor: Option<Arc<dyn Compactor>>,
    /// Checks every response, when configured.
    pub guardrail: Option<Arc<dyn Guardrail>>,
    /// Records what a turn states about the user, when configured.
    pub profile: Option<Arc<ProfileWriter>>,
    /// The profile graph read at assembly time, when configured.
    pub profile_graph: Option<Arc<dyn ProfileGraph>>,
}

/// Records the answer and how the run ended on the span it ran under.
fn record_outcome(span: &tracing::Span, result: &Result<Answer, AgentError>, limit: usize) {
    match result {
        Ok(answer) => {
            let text = truncate(&answer.text, limit);
            span.record("sparky.output", text.as_str());
            span.record("output.value", text.as_str());
            span.record("sparky.status", format!("{:?}", answer.status).as_str());
            let failed = matches!(answer.status, RunStatus::Error | RunStatus::Blocked);
            span.record("otel.status_code", if failed { "ERROR" } else { "OK" });
            if failed {
                span.record(
                    "otel.status_message",
                    format!("{:?}", answer.status).as_str(),
                );
            }
        }
        Err(error) => {
            span.record("otel.status_code", "ERROR");
            span.record("otel.status_message", error.to_string().as_str());
        }
    }
}

/// The agent loop over its dependencies and configuration.
#[derive(Clone)]
pub struct Agent {
    deps: Arc<AgentDeps>,
    cfg: AgentConfig,
    system_prompt: Arc<str>,
    /// Wording written around the prompt sections.
    prompt: Arc<PromptText>,
    /// The capabilities section, rendered once at boot.
    capabilities: Arc<str>,
}

/// What a step decided.
enum StepOutcome {
    /// Keep looping.
    Continue,
    /// Stop with this status and text.
    Stop(RunStatus, String, Option<ConfirmationRequest>),
}

impl Agent {
    /// Builds an agent over its dependencies, with the default prompt wording.
    pub fn new(deps: AgentDeps, cfg: AgentConfig, system_prompt: impl Into<Arc<str>>) -> Self {
        Self {
            deps: Arc::new(deps),
            cfg,
            system_prompt: system_prompt.into(),
            prompt: Arc::new(PromptText::default()),
            capabilities: Arc::from(""),
        }
    }

    /// Replaces the capabilities section the prompt carries.
    pub fn with_capabilities(mut self, capabilities: impl Into<Arc<str>>) -> Self {
        self.capabilities = capabilities.into();
        self
    }

    /// Replaces the wording written around the prompt sections.
    pub fn with_prompt_text(mut self, prompt: PromptText) -> Self {
        self.prompt = Arc::new(prompt);
        self
    }

    /// Runs one user message to completion. One CHAIN span per request, with the conversation
    /// as the session.
    pub async fn run(&self, ctx: &RequestContext, input: &str) -> Result<Answer, AgentError> {
        let asked = truncate(input, self.cfg.max_span_value_chars);
        let span = tracing::info_span!(
            "agent.run",
            "gen_ai.operation.name" = "invoke_agent",
            "gen_ai.agent.name" = "sparky",
            "$ai_session_id" = %ctx.conversation_id,
            "posthog.distinct_id" = %ctx.user_id,
            // OpenInference, read by the Phoenix trace UI.
            "openinference.span.kind" = "CHAIN",
            "input.value" = %asked,
            "output.value" = Empty,
            "session.id" = %ctx.conversation_id,
            "user.id" = %ctx.user_id,
            "sparky.request_id" = %ctx.request_id,
            "sparky.tenant_id" = %ctx.tenant_id,
            "sparky.input" = %asked,
            "sparky.output" = Empty,
            "sparky.status" = Empty,
            "otel.status_code" = Empty,
            "otel.status_message" = Empty,
        );
        let result = self.run_inner(ctx, input).instrument(span.clone()).await;
        record_outcome(&span, &result, self.cfg.max_span_value_chars);
        result
    }

    /// Runs an action the caller approved and carries on to an answer.
    ///
    /// The question and the tool call it produced are already in history. The tool result is
    /// the next turn, and the loop continues from it.
    pub async fn resume(
        &self,
        ctx: &RequestContext,
        pending: PendingAction,
    ) -> Result<Answer, AgentError> {
        let span = tracing::info_span!(
            "agent.resume",
            "gen_ai.operation.name" = "invoke_agent",
            "gen_ai.agent.name" = "sparky",
            "$ai_session_id" = %ctx.conversation_id,
            "posthog.distinct_id" = %ctx.user_id,
            // OpenInference, read by the Phoenix trace UI.
            "openinference.span.kind" = "CHAIN",
            "input.value" = %pending.action.tool,
            "output.value" = Empty,
            "session.id" = %ctx.conversation_id,
            "user.id" = %ctx.user_id,
            "sparky.request_id" = %ctx.request_id,
            "sparky.tenant_id" = %ctx.tenant_id,
            "sparky.input" = %pending.action.tool,
            "sparky.output" = Empty,
            "sparky.status" = Empty,
            "otel.status_code" = Empty,
            "otel.status_message" = Empty,
        );
        let result = self
            .resume_inner(ctx, pending)
            .instrument(span.clone())
            .await;
        record_outcome(&span, &result, self.cfg.max_span_value_chars);
        result
    }

    async fn resume_inner(
        &self,
        ctx: &RequestContext,
        pending: PendingAction,
    ) -> Result<Answer, AgentError> {
        let mut run = Run::new(ctx, "", Vec::new(), 0);
        let call = ToolCall {
            id: pending.call_id,
            name: pending.action.tool,
            arguments: pending.action.arguments,
        };
        let (result, found) = self.run_tool(ctx, 0, &call).await;
        run.tool_sources.extend(found);
        run.tool_runs.push(ToolRun {
            tool: call.name.clone(),
            ok: result.is_ok(),
        });
        let content = result.unwrap_or_else(|error| format!("error: {error}"));
        run.new_turns
            .push(Message::tool_result(&call.id, &call.name, content));

        let inputs = Inputs {
            history: self.history(ctx).await?,
            memory: Vec::new(),
            evidence: Vec::new(),
        };
        self.loop_until_done(&mut run, &inputs).await
    }

    async fn run_inner(&self, ctx: &RequestContext, input: &str) -> Result<Answer, AgentError> {
        let mut run = Run::new(ctx, input, vec![Message::user(input)], 1);
        self.deps.trace.emit(
            ctx,
            TraceEvent::RequestStarted {
                input: input.to_owned(),
                tenant_id: ctx.tenant_id.clone(),
                user_id: ctx.user_id.clone(),
            },
        );

        let inputs = self.load(ctx, input).await?;
        self.loop_until_done(&mut run, &inputs).await
    }

    /// Steps until the loop stops, then keeps the turns and builds the answer.
    async fn loop_until_done(
        &self,
        run: &mut Run<'_>,
        inputs: &Inputs,
    ) -> Result<Answer, AgentError> {
        loop {
            if let Some(stop) = self.check_limits(run) {
                return self
                    .conclude(run, stop, String::new(), inputs.evidence.clone(), None)
                    .await;
            }
            run.steps += 1;

            let (status, text, confirmation) = match self.step(run, inputs).await {
                Ok(StepOutcome::Continue) => continue,
                Ok(StepOutcome::Stop(status, text, confirmation)) => (status, text, confirmation),
                Err(ModelError::Cancelled) => (RunStatus::Cancelled, String::new(), None),
                Err(ModelError::Timeout) => (RunStatus::Deadline, String::new(), None),
                Err(error) => {
                    // The turns of a failed request are still kept.
                    if let Err(kept) = self
                        .conclude(
                            run,
                            RunStatus::Error,
                            String::new(),
                            inputs.evidence.clone(),
                            None,
                        )
                        .await
                    {
                        tracing::error!(error = %kept, "turns of a failed request were not kept");
                    }
                    return Err(error.into());
                }
            };
            return self
                .conclude(run, status, text, inputs.evidence.clone(), confirmation)
                .await;
        }
    }

    /// The status the loop stops with before the next step, if any.
    fn check_limits(&self, run: &Run<'_>) -> Option<RunStatus> {
        if run.ctx.cancel.is_cancelled() {
            Some(RunStatus::Cancelled)
        } else if run.ctx.remaining().is_zero() {
            Some(RunStatus::Deadline)
        } else if run.steps >= self.cfg.max_steps {
            Some(RunStatus::StepLimit)
        } else {
            None
        }
    }

    /// The dollar cost of usage at the configured rates.
    fn cost(&self, usage: Usage) -> f64 {
        (f64::from(usage.prompt_tokens) * self.cfg.usd_per_m_prompt
            + f64::from(usage.completion_tokens) * self.cfg.usd_per_m_completion)
            / 1_000_000.0
    }
}

/// Milliseconds elapsed since since.
fn ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}
