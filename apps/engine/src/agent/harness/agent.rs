//! The agent loop: model call → policy → tool execution → repeat until final answer, error,
//! cancel, deadline, or step limit.

use uuid::Uuid;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use serde_json::Value;
use tracing::Instrument;
use tracing::field::Empty;

use crate::agent::harness::assemble;
use crate::agent::harness::tool::ToolSet;
use crate::core::traits::conversation::ConversationStore;
use crate::core::traits::memory::MemoryStore;
use crate::core::traits::model::ModelProvider;
use crate::core::traits::policy::Policy;
use crate::core::traits::retrieval::Retriever;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::{AgentConfig, AgentError, Answer};
use crate::core::types::assemble::Sections;
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::memory::{Memory, MemoryQuery};
use crate::core::types::message::{Message, ToolCall};
use crate::core::types::model::{FinishReason, ModelError, ModelRequest, ModelResponse, Usage};
use crate::core::types::policy::{ConfirmationRequest, Decision, ProposedAction};
use crate::core::types::retrieval::RetrievalQuery;
use crate::core::types::tool::ToolError;
use crate::core::types::trace::{RunStatus, TraceEvent};

/// Longest value recorded on a span. Phoenix keeps whole values; the JSONL trace has the rest.
const MAX_SPAN_VALUE: usize = 32_000;

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
}

/// The loop. Cheap to clone; holds only `Arc`s.
#[derive(Clone)]
pub struct Agent {
    deps: Arc<AgentDeps>,
    cfg: AgentConfig,
    system_prompt: Arc<str>,
}

/// What one request loaded before its first model call.
struct Inputs {
    history: Vec<Message>,
    memory: Vec<Memory>,
    evidence: Vec<Evidence>,
}

/// Mutable state carried across steps.
struct Run<'a> {
    ctx: &'a RequestContext,
    input: &'a str,
    started: Instant,
    steps: u32,
    usage: Usage,
    /// Turns produced during this request, persisted at the end. First is the user input.
    new_turns: Vec<Message>,
    /// Every (tool, arguments) already executed this request, to catch loops.
    seen_calls: HashSet<String>,
    /// Set after a step of nothing but repeats: the next model call gets no tools, so the
    /// model has to answer from what it already has.
    force_answer: bool,
}

/// What a step decided.
enum StepOutcome {
    /// Keep looping.
    Continue,
    /// Stop with this status and text.
    Stop(RunStatus, String, Option<ConfirmationRequest>),
}

impl Agent {
    /// Builds an agent over its dependencies.
    pub fn new(deps: AgentDeps, cfg: AgentConfig, system_prompt: impl Into<Arc<str>>) -> Self {
        Self {
            deps: Arc::new(deps),
            cfg,
            system_prompt: system_prompt.into(),
        }
    }

    /// Runs one user message to completion. One `CHAIN` span per request, with the
    /// conversation as the session so a Discord thread reads as one session in Phoenix.
    pub async fn run(&self, ctx: &RequestContext, input: &str) -> Result<Answer, AgentError> {
        let span = tracing::info_span!(
            "agent.run",
            "openinference.span.kind" = "CHAIN",
            "session.id" = %ctx.conversation_id,
            "user.id" = %ctx.user_id,
            "sparky.request_id" = %ctx.request_id,
            "sparky.tenant_id" = %ctx.tenant_id,
            "input.value" = %input,
            "output.value" = Empty,
            "sparky.status" = Empty,
        );
        let result = self.run_inner(ctx, input).instrument(span.clone()).await;
        if let Ok(answer) = &result {
            span.record("output.value", truncate(&answer.text, 4_000).as_str());
            span.record("sparky.status", format!("{:?}", answer.status).as_str());
        }
        result
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
            force_answer: false,
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

        loop {
            if let Some(stop) = self.check_limits(&run) {
                return Ok(self.finish(&run, stop, String::new(), inputs.evidence, None));
            }
            run.steps += 1;

            match self.step(&mut run, &inputs).await {
                Ok(StepOutcome::Continue) => {}
                Ok(StepOutcome::Stop(status, text, confirmation)) => {
                    self.persist(ctx, &run.new_turns).await?;
                    return Ok(self.finish(&run, status, text, inputs.evidence, confirmation));
                }
                Err(ModelError::Cancelled) => {
                    return Ok(self.finish(
                        &run,
                        RunStatus::Cancelled,
                        String::new(),
                        inputs.evidence,
                        None,
                    ));
                }
                Err(ModelError::Timeout) => {
                    return Ok(self.finish(
                        &run,
                        RunStatus::Deadline,
                        String::new(),
                        inputs.evidence,
                        None,
                    ));
                }
                Err(error) => {
                    self.finish(&run, RunStatus::Error, String::new(), Vec::new(), None);
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

    async fn load(&self, ctx: &RequestContext, input: &str) -> Result<Inputs, AgentError> {
        let deps = &self.deps;
        let history = match &deps.conversations {
            Some(store) => store
                .load(ctx, self.cfg.history_turns)
                .await
                .map_err(|error| AgentError::Store(error.to_string()))?,
            None => Vec::new(),
        };
        let memory = match &deps.memory {
            Some(store) => store
                .recall(
                    ctx,
                    &MemoryQuery {
                        kinds: Vec::new(),
                        limit: 10,
                    },
                )
                .await
                .map_err(|error| AgentError::Store(error.to_string()))?,
            None => Vec::new(),
        };
        let evidence = match &deps.retriever {
            Some(retriever) => {
                let started = Instant::now();
                let query = RetrievalQuery::new(input, self.cfg.retrieval_top_k);
                let span = tracing::info_span!(
                    "retrieve",
                    "openinference.span.kind" = "RETRIEVER",
                    "input.value" = %input,
                    "output.value" = Empty,
                    "output.mime_type" = "application/json",
                );
                let found = retriever
                    .retrieve(ctx, &query)
                    .instrument(span.clone())
                    .await
                    .map_err(|error| AgentError::Store(format!("retrieval: {error}")))?;
                {
                    let listing: Vec<serde_json::Value> = found
                        .iter()
                        .map(|e| {
                            serde_json::json!({
                                "chunk_id": e.chunk_id,
                                "source_id": e.source_id,
                                "title": e.title,
                                "score": e.score,
                                "content": truncate(&e.content, 1_000),
                            })
                        })
                        .collect();
                    span.record(
                        "output.value",
                        truncate(&json(&listing), MAX_SPAN_VALUE).as_str(),
                    );
                    deps.trace.emit(
                        ctx,
                        TraceEvent::Retrieval {
                            step: 0,
                            query: input.to_owned(),
                            chunk_ids: found.iter().map(|item| item.chunk_id).collect(),
                            duration_ms: ms(started),
                        },
                    );
                    found
                }
            }
            None => Vec::new(),
        };
        Ok(Inputs {
            history,
            memory,
            evidence,
        })
    }

    /// One model call and whatever tool calls it asks for.
    async fn step(&self, run: &mut Run<'_>, inputs: &Inputs) -> Result<StepOutcome, ModelError> {
        let ctx = run.ctx;
        // Prompt history is prior turns plus this request's own turns so far, minus the
        // current input, which assembly appends itself.
        let mut prompt_history = inputs.history.clone();
        prompt_history.extend(run.new_turns.iter().skip(1).cloned());
        // Tool schemas ride along with every request, so they come out of the same budget.
        let tool_tokens: usize = self
            .deps
            .tools
            .definitions()
            .iter()
            .map(|d| (d.name.len() + d.description.len() + d.parameters.to_string().len()) / 4 + 8)
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
                input: run.input,
            },
            budget,
        );
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

        let response = self
            .call_model(ctx, run.steps, assembled.messages, run.force_answer)
            .await?;
        run.usage.add(response.usage);
        run.new_turns.push(response.as_message());

        if response.tool_calls.is_empty() {
            if response.content.trim().is_empty() {
                let (status, text) = if run.force_answer {
                    (
                        RunStatus::Stalled,
                        "I could not turn what I found into an answer. Try rephrasing.".to_owned(),
                    )
                } else if response.finish_reason == FinishReason::Length {
                    (
                        RunStatus::Answered,
                        "I ran out of room before finishing the answer.".to_owned(),
                    )
                } else {
                    (RunStatus::Answered, String::new())
                };
                return Ok(StepOutcome::Stop(status, text, None));
            }
            return Ok(StepOutcome::Stop(
                RunStatus::Answered,
                response.content,
                None,
            ));
        }

        let runnable = match self.authorize_all(run, &response.tool_calls).await {
            Ok(calls) => calls,
            Err(request) => {
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

    /// Runs the calls policy allowed. Repeats are refused and reported; a step made only of
    /// repeats stalls the run. Stateful tools force in-order execution.
    async fn execute(
        &self,
        run: &mut Run<'_>,
        runnable: Vec<ToolCall>,
    ) -> Result<StepOutcome, ModelError> {
        let ctx = run.ctx;
        // A call the model already made with identical arguments is not run again; it is told
        // so. A step made of nothing but repeats means the model is looping.
        let mut fresh = Vec::with_capacity(runnable.len());
        let mut repeats = 0usize;
        for call in runnable {
            let key = format!("{}:{}", call.name, call.arguments);
            if run.seen_calls.insert(key) {
                fresh.push(call);
            } else {
                repeats += 1;
                run.new_turns.push(Message::tool_result(
                    &call.id,
                    &call.name,
                    "already called with these exact arguments earlier in this conversation; \
                     use that result or answer the user",
                ));
            }
        }
        if fresh.is_empty() && repeats > 0 {
            if run.force_answer {
                return Ok(StepOutcome::Stop(
                    RunStatus::Stalled,
                    "I kept repeating the same steps without getting further. Try rephrasing, or \
                     ask for something more specific."
                        .into(),
                    None,
                ));
            }
            run.force_answer = true;
            return Ok(StepOutcome::Continue);
        }

        // Independent calls run in parallel, each under its own timeout. Anything stateful
        // (a browser) forces the whole step to run in order.
        let step = run.steps;
        let stateful = fresh.iter().any(|call| {
            self.deps
                .tools
                .get(&call.name)
                .is_some_and(|t| t.definition().sequential)
        });
        let results: Vec<Result<String, ToolError>> = if stateful {
            let mut out = Vec::with_capacity(fresh.len());
            for call in &fresh {
                out.push(self.run_tool(ctx, step, call).await);
            }
            out
        } else {
            join_all(fresh.iter().map(|call| self.run_tool(ctx, step, call))).await
        };
        for (call, result) in fresh.iter().zip(results) {
            let content = result.unwrap_or_else(|error| format!("error: {error}"));
            run.new_turns
                .push(Message::tool_result(&call.id, &call.name, content));
        }
        Ok(StepOutcome::Continue)
    }

    /// Runs policy over every call before anything executes. Denials are fed back as tool
    /// results; the first confirmation stops the run.
    async fn authorize_all(
        &self,
        run: &mut Run<'_>,
        calls: &[ToolCall],
    ) -> Result<Vec<ToolCall>, ConfirmationRequest> {
        let deps = &self.deps;
        let mut runnable = Vec::new();
        for call in calls {
            let Some(tool) = deps.tools.get(&call.name) else {
                run.new_turns.push(Message::tool_result(
                    &call.id,
                    &call.name,
                    format!("error: no tool named `{}`", call.name),
                ));
                continue;
            };
            let action = ProposedAction {
                tool: call.name.clone(),
                risk: tool.definition().risk,
                arguments: call.arguments.clone(),
            };
            let decision = deps.policy.authorize(run.ctx, &action).await;
            deps.trace.emit(
                run.ctx,
                TraceEvent::PolicyDecision {
                    step: run.steps,
                    tool: call.name.clone(),
                    decision: decision.clone(),
                },
            );
            match decision {
                Decision::Allow => runnable.push(call.clone()),
                Decision::Deny { reason } => {
                    run.new_turns.push(Message::tool_result(
                        &call.id,
                        &call.name,
                        format!("denied: {reason}"),
                    ));
                }
                Decision::Confirm(request) => return Err(request),
            }
        }
        Ok(runnable)
    }

    async fn call_model(
        &self,
        ctx: &RequestContext,
        step: u32,
        messages: Vec<Message>,
        answer_only: bool,
    ) -> Result<ModelResponse, ModelError> {
        let deps = &self.deps;
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
            // Full prompt and full reply as JSON: this is what a training example is made of.
            let input_json = json(&request.messages);
            let span = tracing::info_span!(
                "llm",
                "openinference.span.kind" = "LLM",
                "llm.model_name" = Empty,
                "llm.token_count.prompt" = Empty,
                "llm.token_count.completion" = Empty,
                "llm.invocation_parameters" = %format!(
                    "{{\"max_tokens\":{},\"temperature\":{},\"tools\":{}}}",
                    request.max_tokens,
                    request.temperature,
                    request.tools.len()
                ),
                "input.value" = %truncate(&input_json, MAX_SPAN_VALUE),
                "input.mime_type" = "application/json",
                "output.value" = Empty,
                "output.mime_type" = "application/json",
                "sparky.step" = step,
                "sparky.attempt" = attempt,
            );
            let result = tokio::select! {
                () = ctx.cancel.cancelled() => Err(ModelError::Cancelled),
                outcome = tokio::time::timeout(ctx.remaining(), deps.model.generate(ctx, request).instrument(span.clone())) => {
                    outcome.unwrap_or(Err(ModelError::Timeout))
                }
            };
            match result {
                Ok(response) => {
                    span.record("llm.model_name", response.model.as_str());
                    span.record(
                        "llm.token_count.prompt",
                        i64::from(response.usage.prompt_tokens),
                    );
                    span.record(
                        "llm.token_count.completion",
                        i64::from(response.usage.completion_tokens),
                    );
                    let shown = json(&response.as_message());
                    span.record("output.value", truncate(&shown, MAX_SPAN_VALUE).as_str());
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
                    tokio::time::sleep(backoff(attempt, ctx.request_id, ctx.remaining())).await;
                }
            }
        }
    }

    async fn run_tool(
        &self,
        ctx: &RequestContext,
        step: u32,
        call: &ToolCall,
    ) -> Result<String, ToolError> {
        let deps = &self.deps;
        let Some(tool) = deps.tools.get(&call.name) else {
            return Err(ToolError::Failed(format!("no tool named `{}`", call.name)));
        };
        let started = Instant::now();
        let limit = self.cfg.tool_timeout.min(ctx.remaining());
        let span = tracing::info_span!(
            "tool",
            "openinference.span.kind" = "TOOL",
            "tool.name" = %call.name,
            "tool.call_id" = %call.id,
            "input.value" = %redact(&call.arguments),
            "input.mime_type" = "application/json",
            "output.value" = Empty,
            "sparky.step" = step,
        );
        let result = tokio::select! {
            () = ctx.cancel.cancelled() => Err(ToolError::Cancelled),
            outcome = tokio::time::timeout(limit, tool.call(ctx, call.arguments.clone()).instrument(span.clone())) => {
                outcome.unwrap_or(Err(ToolError::Timeout))
            }
        };
        let (content, traced) = match result {
            Ok(output) => (
                Ok(output.content.clone()),
                Ok(truncate(&output.content, 2_000)),
            ),
            Err(error) => {
                let message = error.to_string();
                (Err(error), Err(message))
            }
        };
        deps.trace.emit(
            ctx,
            TraceEvent::ToolCall {
                step,
                call_id: call.id.clone(),
                tool: call.name.clone(),
                arguments: redact(&call.arguments),
                result: traced,
                duration_ms: ms(started),
            },
        );
        match &content {
            Ok(text) => span.record("output.value", truncate(text, 4_000).as_str()),
            Err(error) => span.record("output.value", format!("error: {error}").as_str()),
        };
        content
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
            usage: run.usage,
            cost_usd,
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

/// Cuts `text` to at most `max` bytes on a char boundary, marking the cut.
pub(crate) fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_owned()
    } else {
        let mut end = max;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &text[..end])
    }
}

/// Drops argument values whose key looks like a secret before they reach the trace.
pub(crate) fn redact(value: &Value) -> Value {
    const SECRET_KEYS: [&str; 6] = [
        "password",
        "token",
        "secret",
        "cookie",
        "authorization",
        "api_key",
    ];
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, inner)| {
                    let lower = key.to_ascii_lowercase();
                    let secret = SECRET_KEYS.iter().any(|needle| lower.contains(needle));
                    (
                        key.clone(),
                        if secret {
                            Value::String("[redacted]".into())
                        } else {
                            redact(inner)
                        },
                    )
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(redact).collect()),
        other => other.clone(),
    }
}

/// JSON for a span attribute. A value that will not serialize is recorded as such rather than
/// as an empty string.
fn json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|e| format!("{{\"unserializable\":{:?}}}", e.to_string()))
}

/// Wait before retry `attempt`: doubling from 250 ms, capped at 8 s, spread by a per-request
/// offset so concurrent requests do not retry in lockstep, and never past the deadline.
pub fn backoff(attempt: u32, request_id: Uuid, remaining: Duration) -> Duration {
    const BASE_MS: u64 = 250;
    const CAP_MS: u64 = 8_000;
    let doubled = BASE_MS.saturating_mul(1u64 << attempt.min(6)).min(CAP_MS);
    let spread = doubled / 4;
    #[allow(clippy::cast_possible_truncation)]
    let offset = (request_id.as_u128() as u64) % spread.max(1);
    Duration::from_millis(doubled - spread / 2 + offset).min(remaining)
}
