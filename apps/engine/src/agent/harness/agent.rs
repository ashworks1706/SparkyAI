//! The agent loop: model call, policy, tool execution, repeated until final answer, error,
//! cancel, deadline, or step limit.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

use futures::future::join_all;
use serde_json::Value;
use tracing::Instrument;
use tracing::field::Empty;

use crate::agent::harness::assemble;
use crate::agent::harness::profile::ProfileWriter;
use crate::agent::harness::tool::ToolSet;
use crate::core::traits::compaction::Compactor;
use crate::core::traits::confirmation::ConfirmationStore;
use crate::core::traits::conversation::ConversationStore;
use crate::core::traits::guardrail::Guardrail;
use crate::core::traits::memory::MemoryStore;
use crate::core::traits::model::ModelProvider;
use crate::core::traits::policy::Policy;
use crate::core::traits::profile::ProfileGraph;
use crate::core::traits::retrieval::Retriever;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::{AgentConfig, AgentError, Answer};
use crate::core::types::assemble::{Sections, Templates};
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::guardrail::{Stage, Verdict};
use crate::core::types::memory::{Memory, MemoryQuery};
use crate::core::types::message::{Message, Role, ToolCall};
use crate::core::types::model::{FinishReason, ModelError, ModelRequest, ModelResponse, Usage};
use crate::core::types::policy::{ConfirmationRequest, Decision, PendingAction, ProposedAction};
use crate::core::types::retrieval::RetrievalQuery;
use crate::core::types::tokens::estimate;
use crate::core::types::tool::{ToolError, ToolRun};
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
    } else if response.finish_reason == FinishReason::Length {
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

/// How many leading turns have to go for the rest to fit budget.
///
/// Counts from the newest backwards, the way assembly spends the budget, and never returns
/// every turn: compacting the whole history leaves the current exchange with no context.
fn overflowing(turns: &[Message], budget: usize, chars_per_token: usize) -> usize {
    let mut spent = 0;
    let mut kept = 0;
    for m in turns.iter().rev() {
        let cost = m.estimated_tokens(chars_per_token);
        if spent + cost > budget {
            break;
        }
        spent += cost;
        kept += 1;
    }
    turns.len().saturating_sub(kept.max(1))
}

/// Why authorize_all stopped.
enum HeldError {
    /// The caller must approve before anything runs.
    Confirm(ConfirmationRequest),
    /// The action could not be held for later approval.
    Store(String),
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

/// The configurable wording assembly writes around the sections.
#[derive(Debug, Clone)]
pub struct PromptText {
    /// Line naming the user, with {user} and {roles}.
    pub role_line: String,
    /// Line naming a user who holds no roles, with {user}.
    pub role_line_no_roles: String,
    /// Heading above recalled memories.
    pub memory_header: String,
    /// Heading above retrieved evidence.
    pub evidence_header: String,
}

impl Default for PromptText {
    fn default() -> Self {
        Self::from(&crate::core::config::Prompt::default())
    }
}

impl From<&crate::core::config::Prompt> for PromptText {
    fn from(cfg: &crate::core::config::Prompt) -> Self {
        Self {
            role_line: cfg.role_line.clone(),
            role_line_no_roles: cfg.role_line_no_roles.clone(),
            memory_header: cfg.memory_header.clone(),
            evidence_header: cfg.evidence_header.clone(),
        }
    }
}

impl PromptText {
    /// Borrowed view for one assembly pass.
    fn templates(&self) -> Templates<'_> {
        Templates {
            role_line: &self.role_line,
            role_line_no_roles: &self.role_line_no_roles,
            memory_header: &self.memory_header,
            evidence_header: &self.evidence_header,
        }
    }
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
    /// Tools that ran, in order, for the answer and the client.
    tool_runs: Vec<ToolRun>,
    /// Set after a step of nothing but repeats. The next model call gets no tools.
    force_answer: bool,
    /// Leading new_turns entries that assembly appends itself, which the prompt must not
    /// repeat. One for a fresh request, none when resuming after an approval.
    appended_by_assembly: usize,
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
            "openinference.span.kind" = "CHAIN",
            "session.id" = %ctx.conversation_id,
            "user.id" = %ctx.user_id,
            "sparky.request_id" = %ctx.request_id,
            "input.value" = %pending.action.tool,
            "output.value" = Empty,
        );
        let result = self
            .resume_inner(ctx, pending)
            .instrument(span.clone())
            .await;
        if let Ok(answer) = &result {
            span.record("output.value", truncate(&answer.text, 4_000).as_str());
        }
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
            force_answer: false,
            appended_by_assembly: 0,
        };
        let call = ToolCall {
            id: pending.call_id,
            name: pending.action.tool,
            arguments: pending.action.arguments,
        };
        let result = self.run_tool(ctx, 0, &call).await;
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

    async fn history(&self, ctx: &RequestContext) -> Result<Vec<Message>, AgentError> {
        let Some(store) = &self.deps.conversations else {
            return Ok(Vec::new());
        };
        let loaded = store
            .load(ctx, self.cfg.history_turns)
            .await
            .map_err(|error| AgentError::Store(error.to_string()))?;
        Ok(self.compacted(ctx, loaded).await)
    }

    /// Replaces the turns that do not fit the history budget with one compacted turn, and keeps
    /// it so the next request starts from it.
    ///
    /// A compaction that fails leaves the history trimmed the way it always was.
    async fn compacted(&self, ctx: &RequestContext, turns: Vec<Message>) -> Vec<Message> {
        let Some(compactor) = &self.deps.compactor else {
            return turns;
        };
        let cpt = self.cfg.budget.chars_per_token;
        let overflow = overflowing(&turns, self.cfg.budget.history, cpt);
        if overflow == 0 {
            return turns;
        }
        let (replaced, kept) = turns.split_at(overflow);
        self.deps.trace.emit(
            ctx,
            TraceEvent::Compaction {
                turns: replaced.len(),
            },
        );
        let summary = match compactor.compact(ctx, replaced).await {
            Ok(summary) => summary,
            Err(error) => {
                tracing::warn!(error = %error, "compaction failed; history is trimmed instead");
                return turns;
            }
        };
        if let Some(store) = &self.deps.conversations
            && let Err(error) = store.append(ctx, std::slice::from_ref(&summary)).await
        {
            // The prompt still gets the summary. Only the saving of it failed.
            tracing::warn!(error = %error, "compacted turn was not stored");
        }
        let mut out = Vec::with_capacity(kept.len() + 1);
        out.push(summary);
        out.extend_from_slice(kept);
        out
    }

    async fn load(&self, ctx: &RequestContext, input: &str) -> Result<Inputs, AgentError> {
        let deps = &self.deps;
        let history = self.history(ctx).await?;
        let memory = match &deps.memory {
            Some(store) => store
                .recall(
                    ctx,
                    &MemoryQuery {
                        kinds: Vec::new(),
                        limit: self.cfg.memory_recall_limit,
                    },
                )
                .await
                .map_err(|error| AgentError::Store(error.to_string()))?,
            None => Vec::new(),
        };
        let memory = self.with_profile(ctx, memory).await;
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
                        truncate(&json(&listing), self.cfg.max_span_value_chars).as_str(),
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
                templates: self.prompt.templates(),
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

        let stage = if response.tool_calls.is_empty() {
            Stage::Answer
        } else {
            Stage::Capability
        };
        if let Some(blocked) = self.guarded(ctx, run.steps, stage, &response.content).await {
            return Ok(blocked);
        }

        if response.tool_calls.is_empty() {
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

    /// Runs the calls policy allowed. Repeats are refused and reported, and a step made only
    /// of repeats stalls the run. Stateful tools force in-order execution.
    async fn execute(
        &self,
        run: &mut Run<'_>,
        runnable: Vec<ToolCall>,
    ) -> Result<StepOutcome, ModelError> {
        let ctx = run.ctx;
        // A call the model already made with identical arguments is not run again. The model
        // is told so.
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
        // forces the whole step to run in order.
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
            run.tool_runs.push(ToolRun {
                tool: call.name.clone(),
                ok: result.is_ok(),
            });
            let content = result.unwrap_or_else(|error| format!("error: {error}"));
            run.new_turns
                .push(Message::tool_result(&call.id, &call.name, content));
        }
        Ok(StepOutcome::Continue)
    }

    /// Runs policy over every call before anything executes. Denials are fed back as tool
    /// results, and the first confirmation stops the run.
    async fn authorize_all(
        &self,
        run: &mut Run<'_>,
        calls: &[ToolCall],
    ) -> Result<Vec<ToolCall>, HeldError> {
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
                Decision::Confirm(request) => {
                    let pending = PendingAction {
                        call_id: call.id.clone(),
                        action,
                    };
                    if let Some(store) = &deps.confirmations {
                        store
                            .hold(
                                run.ctx,
                                request.token,
                                &pending,
                                &request.payload_hash,
                                self.cfg.confirmation_ttl,
                            )
                            .await
                            .map_err(|error| HeldError::Store(error.to_string()))?;
                    }
                    return Err(HeldError::Confirm(request));
                }
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
            // Full prompt and full reply as JSON.
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
                "input.value" = %truncate(&input_json, self.cfg.max_span_value_chars),
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
                    span.record(
                        "output.value",
                        truncate(&shown, self.cfg.max_span_value_chars).as_str(),
                    );
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
        deps.trace.emit(
            ctx,
            TraceEvent::ToolStarted {
                step,
                tool: call.name.clone(),
            },
        );
        let started = Instant::now();
        // A tool may declare its own budget. The request deadline still wins.
        let declared = tool
            .definition()
            .timeout_secs
            .map_or(self.cfg.tool_timeout, Duration::from_secs);
        let limit = declared.min(ctx.remaining());
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

    /// Keeps the turns, records the outcome, and builds the answer. Every exit from the loop
    /// goes through here.
    async fn conclude(
        &self,
        run: &Run<'_>,
        status: RunStatus,
        text: String,
        evidence: Vec<Evidence>,
        confirmation: Option<ConfirmationRequest>,
    ) -> Result<Answer, AgentError> {
        self.persist(run.ctx, &run.new_turns).await?;
        self.record_profile(run);
        let text = if text.trim().is_empty() {
            status.explain().unwrap_or_default().to_owned()
        } else {
            text
        };
        Ok(self.finish(run, status, text, evidence, confirmation))
    }

    /// Appends what the graph knows about this user to the memories recalled. A graph that
    /// cannot be read leaves the prompt with the memories alone.
    async fn with_profile(&self, ctx: &RequestContext, mut memory: Vec<Memory>) -> Vec<Memory> {
        let Some(graph) = &self.deps.profile_graph else {
            return memory;
        };
        match graph.recall(ctx, self.cfg.memory_recall_limit).await {
            Ok(nodes) => memory.extend(nodes.iter().map(Memory::from)),
            Err(error) => tracing::warn!(error = %error, "profile graph recall failed"),
        }
        memory
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

/// Cuts text to at most max bytes on a char boundary, marking the cut.
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

/// JSON for a span attribute. A value that will not serialize is recorded as unserializable.
fn json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_else(|e| format!("{{\"unserializable\":{:?}}}", e.to_string()))
}

/// Wait before retry attempt: doubling from base_ms, capped at cap_ms, spread by a per-request
/// offset, and never past the deadline.
pub fn backoff(
    attempt: u32,
    request_id: Uuid,
    remaining: Duration,
    base_ms: u64,
    cap_ms: u64,
) -> Duration {
    let doubled = base_ms.saturating_mul(1u64 << attempt.min(6)).min(cap_ms);
    let spread = doubled / 4;
    #[allow(clippy::cast_possible_truncation)]
    let offset = (request_id.as_u128() as u64) % spread.max(1);
    Duration::from_millis(doubled - spread / 2 + offset).min(remaining)
}
