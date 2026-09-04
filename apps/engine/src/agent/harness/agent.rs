//! The agent loop: model call → policy → tool execution → repeat until final answer, error,
//! cancel, deadline, or step limit.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use serde_json::Value;

use crate::agent::harness::assemble;
use crate::agent::harness::conversation::ConversationStore;
use crate::agent::harness::memory::MemoryStore;
use crate::agent::harness::model::ModelProvider;
use crate::agent::harness::policy::Policy;
use crate::agent::harness::retrieval::Retriever;
use crate::agent::harness::tool::ToolSet;
use crate::agent::harness::trace::TraceSink;
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

    /// Runs one user message to completion.
    pub async fn run(&self, ctx: &RequestContext, input: &str) -> Result<Answer, AgentError> {
        let mut run = Run {
            ctx,
            input,
            started: Instant::now(),
            steps: 0,
            usage: Usage::default(),
            new_turns: vec![Message::user(input)],
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
                    self.persist(ctx, &run.new_turns).await;
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
                .unwrap_or_default(),
            None => Vec::new(),
        };
        let evidence = match &deps.retriever {
            Some(retriever) => {
                let started = Instant::now();
                let query = RetrievalQuery::new(input, self.cfg.retrieval_top_k);
                match retriever.retrieve(ctx, &query).await {
                    Ok(found) => {
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
                    Err(error) => {
                        tracing::warn!(error = %error, "retrieval failed; answering without evidence");
                        Vec::new()
                    }
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
        let assembled = assemble::assemble(
            ctx,
            &Sections {
                system: &self.system_prompt,
                memory: &inputs.memory,
                evidence: &inputs.evidence,
                history: &prompt_history,
                input: run.input,
            },
            self.cfg.budget,
        );
        self.deps.trace.emit(
            ctx,
            TraceEvent::ContextAssembled {
                step: run.steps,
                message_count: assembled.messages.len(),
                estimated_tokens: assembled.estimated_tokens,
                evidence_ids: inputs
                    .evidence
                    .iter()
                    .take(assembled.evidence_used)
                    .map(|item| item.chunk_id)
                    .collect(),
            },
        );

        let response = self.call_model(ctx, run.steps, assembled.messages).await?;
        run.usage.add(response.usage);
        run.new_turns.push(response.as_message());

        if response.tool_calls.is_empty() {
            let text = if response.content.trim().is_empty()
                && response.finish_reason == FinishReason::Length
            {
                "I ran out of room before finishing the answer.".to_owned()
            } else {
                response.content
            };
            return Ok(StepOutcome::Stop(RunStatus::Answered, text, None));
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

        // Independent calls run in parallel, each under its own timeout.
        let step = run.steps;
        let results = join_all(runnable.iter().map(|call| self.run_tool(ctx, step, call))).await;
        for (call, result) in runnable.iter().zip(results) {
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
            let decision = deps
                .policy
                .authorize(run.ctx, &action)
                .await
                .unwrap_or_else(|error| Decision::Deny {
                    reason: format!("policy unavailable: {error}"),
                });
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
    ) -> Result<ModelResponse, ModelError> {
        let deps = &self.deps;
        let mut attempt = 0u32;
        loop {
            let request = ModelRequest {
                messages: messages.clone(),
                tools: deps.tools.definitions(),
                max_tokens: self.cfg.max_tokens,
                temperature: self.cfg.temperature,
            };
            let started = Instant::now();
            let result = tokio::select! {
                () = ctx.cancel.cancelled() => Err(ModelError::Cancelled),
                outcome = tokio::time::timeout(ctx.remaining(), deps.model.generate(ctx, request)) => {
                    outcome.unwrap_or(Err(ModelError::Timeout))
                }
            };
            match result {
                Ok(response) => {
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
                    let backoff = Duration::from_millis(200 * u64::from(attempt));
                    tokio::time::sleep(backoff.min(ctx.remaining())).await;
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
        let result = tokio::select! {
            () = ctx.cancel.cancelled() => Err(ToolError::Cancelled),
            outcome = tokio::time::timeout(limit, tool.call(ctx, call.arguments.clone())) => {
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
        content
    }

    async fn persist(&self, ctx: &RequestContext, turns: &[Message]) {
        if let Some(store) = &self.deps.conversations
            && let Err(error) = store.append(ctx, turns).await
        {
            tracing::warn!(error = %error, "failed to persist turns");
        }
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
