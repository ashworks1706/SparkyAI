//! The agent loop: model call, policy, tool execution, repeated until final answer, error,
//! cancel, deadline, or step limit.

mod conclude;
mod execute;
mod inputs;
pub mod prompt;
mod retry;
mod run;
mod spans;
pub mod task;
pub mod thought;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use tracing::Instrument;
use tracing::field::Empty;

use self::execute::HeldError;
use crate::agent::harness::agent::prompt::assemble;
use crate::agent::harness::agent::run::{Inputs, Run};
use crate::agent::harness::memory::profile::ProfileWriter;
use crate::agent::harness::safety::redact::json;
use crate::agent::harness::tools::ToolSet;
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
use crate::core::types::agent::assemble::Sections;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::agent::{AgentConfig, AgentError, Answer};
use crate::core::types::conversation::message::{Message, ToolCall};
use crate::core::types::model::tokens::estimate;
use crate::core::types::model::{FinishReason, ModelError, ModelRequest, ModelResponse, Usage};
use crate::core::types::safety::guardrail::{Stage, Verdict};
use crate::core::types::safety::policy::{ConfirmationRequest, PendingAction};
use crate::core::types::tools::ToolRun;
use crate::core::types::trace::{RunStatus, TraceEvent};

pub use crate::agent::harness::agent::prompt::PromptText;
pub use crate::agent::harness::agent::retry::backoff;
use crate::agent::harness::safety::redact::truncate;

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

/// The outcome of a step that asked for no capabilities.
fn answered(response: &ModelResponse, force_answer: bool) -> StepOutcome {
    if !response.content.trim().is_empty() {
        return StepOutcome::Stop(RunStatus::Answered, response.content.clone(), None);
    }
    let (status, text) = if force_answer {
        (
            RunStatus::Stalled,
            "I could not turn what I found into an answer. Try rephrasing.",
        )
    } else if response.finish_reason == FinishReason::Length
        || !response.reasoning.trim().is_empty()
    {
        // Reasoning that leaves no room for an answer spends the whole completion budget.
        (
            RunStatus::Answered,
            "I ran out of room before finishing the answer.",
        )
    } else {
        (
            RunStatus::Answered,
            "The model returned nothing. Try rephrasing.",
        )
    };
    StepOutcome::Stop(status, text.to_owned(), None)
}

/// The loop. Cheap to clone, holding only Arcs.
#[derive(Clone)]
pub struct Agent {
    deps: Arc<AgentDeps>,
    cfg: AgentConfig,
    system_prompt: Arc<str>,
    /// Wording written around the prompt sections. Owned, and sourced from configuration.
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
        let mut run = Run {
            ctx,
            input: "",
            started: Instant::now(),
            steps: 0,
            usage: Usage::default(),
            new_turns: Vec::new(),
            seen_calls: HashSet::new(),
            tool_runs: Vec::new(),
            evidence_in_prompt: 0,
            memories_in_prompt: Vec::new(),
            tool_evidence: Vec::new(),
            force_answer: false,
            appended_by_assembly: 0,
        };
        let call = ToolCall {
            id: pending.call_id,
            name: pending.action.tool,
            arguments: pending.action.arguments,
        };
        let (result, found) = self.run_tool(ctx, 0, &call).await;
        run.tool_evidence.extend(found);
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
        let mut run = Run {
            ctx,
            input,
            started: Instant::now(),
            steps: 0,
            usage: Usage::default(),
            new_turns: vec![Message::user(input)],
            seen_calls: HashSet::new(),
            tool_runs: Vec::new(),
            evidence_in_prompt: 0,
            memories_in_prompt: Vec::new(),
            tool_evidence: Vec::new(),
            force_answer: false,
            appended_by_assembly: 1,
        };
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

            match self.step(run, inputs).await {
                Ok(StepOutcome::Continue) => {}
                Ok(StepOutcome::Stop(status, text, confirmation)) => {
                    return self
                        .conclude(run, status, text, inputs.evidence.clone(), confirmation)
                        .await;
                }
                Err(ModelError::Cancelled) => {
                    return self
                        .conclude(
                            run,
                            RunStatus::Cancelled,
                            String::new(),
                            inputs.evidence.clone(),
                            None,
                        )
                        .await;
                }
                Err(ModelError::Timeout) => {
                    return self
                        .conclude(
                            run,
                            RunStatus::Deadline,
                            String::new(),
                            inputs.evidence.clone(),
                            None,
                        )
                        .await;
                }
                Err(error) => {
                    // The turns are still kept. A failed request is part of the conversation.
                    let _ = self
                        .conclude(
                            run,
                            RunStatus::Error,
                            String::new(),
                            inputs.evidence.clone(),
                            None,
                        )
                        .await;
                    return Err(error.into());
                }
            }
        }
    }

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

    /// One model call and whatever tool calls it asks for.
    async fn step(&self, run: &mut Run<'_>, inputs: &Inputs) -> Result<StepOutcome, ModelError> {
        let ctx = run.ctx;
        // Prompt history is prior turns plus the turns of this request so far, minus the
        // current input, which assembly appends itself.
        let mut prompt_history = inputs.history.clone();
        prompt_history.extend(run.new_turns.iter().skip(run.appended_by_assembly).cloned());
        let cpt = self.cfg.budget.chars_per_token;
        // Tool schemas ride along with every request and come out of the same budget.
        let tool_tokens: usize = self
            .deps
            .tools
            .definitions()
            .iter()
            .map(|d| {
                let schema = d.parameters.to_string();
                estimate(&d.name, cpt) + estimate(&d.description, cpt) + estimate(&schema, cpt)
            })
            .sum();
        let mut budget = self.cfg.budget;
        budget.total = budget.total.saturating_sub(tool_tokens);
        let assembled = assemble::assemble(
            ctx,
            &Sections {
                system: &self.system_prompt,
                memory: &inputs.memory,
                evidence: &inputs.evidence,
                history: &prompt_history,
                capabilities: &self.capabilities,
                input: run.input,
                date: &self.prompt.today(),
                templates: self.prompt.templates(),
            },
            budget,
        );
        run.evidence_in_prompt = assembled.evidence_used;
        run.memories_in_prompt = inputs
            .memory
            .iter()
            .take(assembled.memory_used)
            .map(|m| m.content.clone())
            .collect();
        self.deps.trace.emit(
            ctx,
            TraceEvent::ContextAssembled {
                step: run.steps,
                message_count: assembled.messages.len(),
                estimated_tokens: assembled.estimated_tokens + tool_tokens,
                evidence_ids: inputs
                    .evidence
                    .iter()
                    .take(assembled.evidence_used)
                    .map(|item| item.chunk_id)
                    .collect(),
            },
        );

        let mut response = self
            .call_model(ctx, run.steps, assembled.messages, run.force_answer)
            .await?;
        run.usage.add(response.usage);
        // Thinking the model wrote inline belongs to the trace, not to the answer or the
        // history.
        let (thought, visible) = thought::split(&response.reasoning, &response.content);
        response.content = visible;
        run.new_turns.push(response.as_message());

        let thought_shown = self.report_thought(ctx, run.steps, thought, &response);

        let stage = if response.tool_calls.is_empty() {
            Stage::Answer
        } else {
            Stage::Capability
        };
        if let Some(blocked) = self.guarded(ctx, run.steps, stage, &response.content).await {
            return Ok(blocked);
        }

        if response.tool_calls.is_empty() {
            // Replaces the thinking line of a step that thought nothing worth showing. A step
            // that did keeps the thought.
            if !thought_shown {
                self.deps
                    .trace
                    .emit(ctx, TraceEvent::ModelAnswered { step: run.steps });
            }
            return Ok(answered(&response, run.force_answer));
        }

        let runnable = match self.authorize_all(run, &response.tool_calls).await {
            Ok(calls) => calls,
            Err(HeldError::Store(error)) => {
                // An action that cannot be approved later is not offered for approval.
                return Err(ModelError::Transport(format!(
                    "could not hold the action for approval: {error}"
                )));
            }
            Err(HeldError::Confirm(request)) => {
                let text = format!("Before I do that, please confirm: {}", request.summary);
                return Ok(StepOutcome::Stop(
                    RunStatus::AwaitingConfirmation,
                    text,
                    Some(request),
                ));
            }
        };

        self.execute(run, runnable).await
    }

    /// Reports what the model was thinking on this step, and says whether anything was shown.
    ///
    /// The model's own reasoning when it returned some, and otherwise what it wrote on its way
    /// to a tool call.
    fn report_thought(
        &self,
        ctx: &RequestContext,
        step: u32,
        reasoned: Option<String>,
        response: &ModelResponse,
    ) -> bool {
        let shown = reasoned.or_else(|| {
            let preamble = response.content.trim();
            (!response.tool_calls.is_empty() && !preamble.is_empty()).then(|| preamble.to_owned())
        });
        let Some(text) = shown else {
            return false;
        };
        self.deps.trace.emit(
            ctx,
            TraceEvent::ModelThought {
                step,
                text: truncate(&text, self.cfg.max_span_value_chars),
            },
        );
        true
    }

    /// Checks a response against the guardrail. A block ends the run with the replacement text.
    async fn guarded(
        &self,
        ctx: &RequestContext,
        step: u32,
        stage: Stage,
        text: &str,
    ) -> Option<StepOutcome> {
        let guardrail = self.deps.guardrail.as_ref()?;
        // An empty capability branch has no text to check and every step would pay for it.
        if stage == Stage::Capability && text.trim().is_empty() {
            return None;
        }
        let verdict = guardrail.check(ctx, stage, text).await;
        let Verdict::Block {
            replacement,
            reason,
        } = verdict
        else {
            return None;
        };
        self.deps.trace.emit(
            ctx,
            TraceEvent::GuardrailBlocked {
                step,
                stage,
                reason,
            },
        );
        Some(StepOutcome::Stop(RunStatus::Blocked, replacement, None))
    }

    /// The span of one model call. Full prompt and full reply as JSON, under the gen_ai names
    /// PostHog reads and the OpenInference names the Phoenix trace UI reads.
    fn model_span(
        &self,
        ctx: &RequestContext,
        step: u32,
        attempt: u32,
        request: &ModelRequest,
    ) -> tracing::Span {
        let limit = self.cfg.max_span_value_chars;
        let prompt = truncate(&json(&request.messages), limit);
        tracing::info_span!(
            "llm",
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
            "sparky.span" = "llm",
            "sparky.tools" = %truncate(&json(&request.tools), limit),
            "sparky.step" = step,
            "sparky.attempt" = attempt,
            "$ai_session_id" = %ctx.conversation_id,
            "posthog.distinct_id" = %ctx.user_id,
            "otel.status_code" = Empty,
            "otel.status_message" = Empty,
        )
    }

    async fn call_model(
        &self,
        ctx: &RequestContext,
        step: u32,
        messages: Vec<Message>,
        answer_only: bool,
    ) -> Result<ModelResponse, ModelError> {
        let deps = &self.deps;
        deps.trace.emit(ctx, TraceEvent::ModelStarted { step });
        let mut attempt = 0u32;
        loop {
            let request = ModelRequest {
                messages: messages.clone(),
                tools: if answer_only {
                    Vec::new()
                } else {
                    deps.tools.definitions()
                },
                max_tokens: self.cfg.max_tokens,
                temperature: self.cfg.temperature,
            };
            let started = Instant::now();
            let limit = self.cfg.max_span_value_chars;
            let span = self.model_span(ctx, step, attempt, &request);
            let result = tokio::select! {
                () = ctx.cancel.cancelled() => Err(ModelError::Cancelled),
                outcome = tokio::time::timeout(ctx.remaining(), deps.model.generate(ctx, request).instrument(span.clone())) => {
                    outcome.unwrap_or(Err(ModelError::Timeout))
                }
            };
            match result {
                Ok(response) => {
                    span.record("otel.status_code", "OK");
                    spans::record_reply(&span, &response, limit);
                    deps.trace.emit(
                        ctx,
                        TraceEvent::ModelCall {
                            step,
                            model: response.model.clone(),
                            finish_reason: response.finish_reason,
                            usage: response.usage,
                            duration_ms: ms(started),
                            attempt,
                        },
                    );
                    return Ok(response);
                }
                Err(error) => {
                    span.record("otel.status_code", "ERROR");
                    span.record("otel.status_message", error.to_string().as_str());
                    let retry = error.is_retryable() && attempt < self.cfg.max_model_retries;
                    deps.trace.emit(
                        ctx,
                        TraceEvent::ModelError {
                            step,
                            attempt,
                            error: error.to_string(),
                            retried: retry,
                        },
                    );
                    if !retry {
                        return Err(error);
                    }
                    attempt += 1;
                    tokio::time::sleep(backoff(
                        attempt,
                        ctx.request_id,
                        ctx.remaining(),
                        self.cfg.retry_base_ms,
                        self.cfg.retry_cap_ms,
                    ))
                    .await;
                }
            }
        }
    }

    fn cost(&self, usage: Usage) -> f64 {
        (f64::from(usage.prompt_tokens) * self.cfg.usd_per_m_prompt
            + f64::from(usage.completion_tokens) * self.cfg.usd_per_m_completion)
            / 1_000_000.0
    }
}

fn ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}
