//! The agent loop: model call → policy → tool execution → repeat until final answer, error,
//! cancel, deadline, or step limit.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::harness::assemble::{self, Budget, Sections};
use crate::agent::harness::conversation::ConversationStore;
use crate::agent::harness::memory::{Memory, MemoryQuery, MemoryStore};
use crate::agent::harness::model::{
    FinishReason, ModelError, ModelProvider, ModelRequest, ModelResponse, Usage,
};
use crate::agent::harness::policy::{ConfirmationRequest, Decision, Policy, ProposedAction};
use crate::agent::harness::retrieval::{RetrievalQuery, Retriever};
use crate::agent::harness::tool::{ToolError, ToolSet};
use crate::agent::harness::trace::{RunStatus, TraceEvent, TraceSink};
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::message::{Message, ToolCall};

/// Knobs for the loop. All bounded; nothing runs forever.
#[derive(Debug, Clone, Copy)]
pub struct AgentConfig {
    /// Maximum model calls per request.
    pub max_steps: u32,
    /// Retries on a retryable model error, per step.
    pub max_model_retries: u32,
    /// Per-tool-call timeout.
    pub tool_timeout: Duration,
    /// Completion budget per model call.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Evidence chunks to retrieve per request.
    pub retrieval_top_k: usize,
    /// Prior turns to load.
    pub history_turns: usize,
    /// USD per million prompt tokens, for cost tracking. Zero for local models.
    pub usd_per_m_prompt: f64,
    /// USD per million completion tokens.
    pub usd_per_m_completion: f64,
    /// Prompt budgets.
    pub budget: Budget,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_model_retries: 2,
            tool_timeout: Duration::from_secs(20),
            max_tokens: 1024,
            temperature: 0.3,
            retrieval_top_k: 6,
            history_turns: 20,
            usd_per_m_prompt: 0.0,
            usd_per_m_completion: 0.0,
            budget: Budget::default(),
        }
    }
}

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

/// How a run ended and what it produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    /// Final text. Empty when awaiting confirmation.
    pub text: String,
    /// Evidence the answer was grounded in, best first.
    pub evidence: Vec<Evidence>,
    /// Set when the loop stopped to ask the user.
    pub confirmation: Option<ConfirmationRequest>,
    /// How it ended.
    pub status: RunStatus,
    /// Model calls made.
    pub steps: u32,
    /// Tokens across every call.
    pub usage: Usage,
    /// Estimated cost in USD.
    pub cost_usd: f64,
}

/// Loop failures. Everything recoverable has already been fed back to the model.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// The model failed after retries.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A store the request needs was unavailable.
    #[error("store: {0}")]
    Store(String),
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
            run.new_turns.push(Message::tool_result(&call.id, content));
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
                    run.new_turns
                        .push(Message::tool_result(&call.id, format!("denied: {reason}")));
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

fn truncate(text: &str, max: usize) -> String {
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
fn redact(value: &Value) -> Value {
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::agent::harness::policy::RiskPolicy;
    use crate::agent::harness::tool::{RiskClass, Tool, ToolDefinition, ToolOutput};
    use crate::agent::harness::trace::MemorySink;

    /// Replays canned responses in order. Lives in tests only.
    struct Scripted(Mutex<Vec<Result<ModelResponse, ModelError>>>);

    impl Scripted {
        fn new(items: Vec<Result<ModelResponse, ModelError>>) -> Self {
            let mut reversed = items;
            reversed.reverse();
            Self(Mutex::new(reversed))
        }
    }

    #[async_trait]
    impl ModelProvider for Scripted {
        async fn generate(
            &self,
            _ctx: &RequestContext,
            _req: ModelRequest,
        ) -> Result<ModelResponse, ModelError> {
            self.0
                .lock()
                .ok()
                .and_then(|mut items| items.pop())
                .unwrap_or_else(|| Err(ModelError::Malformed("script exhausted".into())))
        }
    }

    fn text(content: &str) -> ModelResponse {
        ModelResponse {
            content: content.into(),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
            model: "test".into(),
        }
    }

    fn calls(items: Vec<(&str, &str, Value)>) -> ModelResponse {
        ModelResponse {
            content: String::new(),
            tool_calls: items
                .into_iter()
                .map(|(id, name, arguments)| ToolCall {
                    id: id.into(),
                    name: name.into(),
                    arguments,
                })
                .collect(),
            finish_reason: FinishReason::ToolCalls,
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
            model: "test".into(),
        }
    }

    struct Echo(RiskClass);

    #[async_trait]
    impl Tool for Echo {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".into(),
                description: "echoes".into(),
                parameters: json!({"type": "object"}),
                risk: self.0,
            }
        }
        async fn call(&self, _ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text(args.to_string()))
        }
    }

    struct Slow;

    #[async_trait]
    impl Tool for Slow {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "slow".into(),
                description: "sleeps".into(),
                parameters: json!({"type": "object"}),
                risk: RiskClass::ReadPublic,
            }
        }
        async fn call(&self, _ctx: &RequestContext, _args: Value) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(Duration::from_secs(5)).await;
            Ok(ToolOutput::text("late"))
        }
    }

    fn agent(model: Scripted, tools: ToolSet, cfg: AgentConfig) -> (Agent, Arc<MemorySink>) {
        let sink = Arc::new(MemorySink::default());
        let deps = AgentDeps {
            model: Arc::new(model),
            tools,
            policy: Arc::new(RiskPolicy::new(Some("Moderator".into()))),
            trace: sink.clone(),
            retriever: None,
            conversations: None,
            memory: None,
        };
        (Agent::new(deps, cfg, "sys"), sink)
    }

    fn ctx() -> RequestContext {
        RequestContext::new("g", "u", Duration::from_secs(5))
    }

    #[tokio::test]
    async fn text_reply_is_the_answer() {
        let (agent, sink) = agent(
            Scripted::new(vec![Ok(text("2am"))]),
            ToolSet::new(),
            AgentConfig::default(),
        );
        let out = agent.run(&ctx(), "when?").await.ok();
        let out = out.as_ref();
        assert_eq!(out.map(|answer| answer.text.as_str()), Some("2am"));
        assert_eq!(out.map(|answer| answer.steps), Some(1));
        assert!(
            sink.records()
                .iter()
                .any(|record| matches!(record.event, TraceEvent::Completed { .. }))
        );
    }

    #[tokio::test]
    async fn tool_result_is_fed_back_and_loop_continues() {
        let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
        let (agent, sink) = agent(
            Scripted::new(vec![
                Ok(calls(vec![("c1", "echo", json!({"q": 1}))])),
                Ok(text("done")),
            ]),
            tools,
            AgentConfig::default(),
        );
        let out = agent.run(&ctx(), "go").await.ok();
        assert_eq!(
            out.as_ref().map(|answer| answer.text.as_str()),
            Some("done")
        );
        assert_eq!(out.as_ref().map(|answer| answer.steps), Some(2));
        assert!(sink.records().iter().any(
            |record| matches!(&record.event, TraceEvent::ToolCall { tool, .. } if tool == "echo")
        ));
    }

    #[tokio::test]
    async fn parallel_calls_all_run() {
        let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
        let (agent, sink) = agent(
            Scripted::new(vec![
                Ok(calls(vec![
                    ("c1", "echo", json!(1)),
                    ("c2", "echo", json!(2)),
                    ("c3", "echo", json!(3)),
                ])),
                Ok(text("ok")),
            ]),
            tools,
            AgentConfig::default(),
        );
        let _ = agent.run(&ctx(), "go").await;
        let count = sink
            .records()
            .iter()
            .filter(|record| matches!(record.event, TraceEvent::ToolCall { .. }))
            .count();
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn write_without_role_is_denied_not_run() {
        let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ExternalWrite)));
        let (agent, sink) = agent(
            Scripted::new(vec![
                Ok(calls(vec![("c1", "echo", json!({}))])),
                Ok(text("ok")),
            ]),
            tools,
            AgentConfig::default(),
        );
        let _ = agent.run(&ctx(), "post it").await;
        let records = sink.records();
        assert!(records.iter().any(|record| matches!(
            &record.event,
            TraceEvent::PolicyDecision {
                decision: Decision::Deny { .. },
                ..
            }
        )));
        assert!(
            !records
                .iter()
                .any(|record| matches!(record.event, TraceEvent::ToolCall { .. }))
        );
    }

    #[tokio::test]
    async fn write_with_role_stops_for_confirmation() {
        let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ExternalWrite)));
        let (agent, _) = agent(
            Scripted::new(vec![
                Ok(calls(vec![("c1", "echo", json!({}))])),
                Ok(text("never")),
            ]),
            tools,
            AgentConfig::default(),
        );
        let context = ctx().with_roles(vec!["Moderator".into()]);
        let out = agent.run(&context, "post it").await.ok();
        assert_eq!(
            out.as_ref().map(|answer| answer.status.clone()),
            Some(RunStatus::AwaitingConfirmation)
        );
        assert!(
            out.as_ref()
                .is_some_and(|answer| answer.confirmation.is_some())
        );
    }

    #[tokio::test]
    async fn step_limit_stops_the_loop() {
        let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
        let script: Vec<_> = (0..10)
            .map(|i| Ok(calls(vec![(&format!("c{i}"), "echo", json!(i))])))
            .collect();
        let (agent, _) = agent(
            Scripted::new(script),
            tools,
            AgentConfig {
                max_steps: 3,
                ..AgentConfig::default()
            },
        );
        let out = agent.run(&ctx(), "loop").await.ok();
        assert_eq!(
            out.as_ref().map(|answer| answer.status.clone()),
            Some(RunStatus::StepLimit)
        );
        assert_eq!(out.as_ref().map(|answer| answer.steps), Some(3));
    }

    #[tokio::test]
    async fn tool_timeout_becomes_an_error_result() {
        let tools = ToolSet::new().with(Arc::new(Slow));
        let (agent, sink) = agent(
            Scripted::new(vec![
                Ok(calls(vec![("c1", "slow", json!({}))])),
                Ok(text("ok")),
            ]),
            tools,
            AgentConfig {
                tool_timeout: Duration::from_millis(50),
                ..AgentConfig::default()
            },
        );
        let _ = agent.run(&ctx(), "go").await;
        assert!(sink.records().iter().any(|record| matches!(
            &record.event,
            TraceEvent::ToolCall { result: Err(message), .. } if message.contains("timed out")
        )));
    }

    #[tokio::test]
    async fn cancellation_ends_the_run() {
        let (agent, _) = agent(
            Scripted::new(vec![Ok(text("x"))]),
            ToolSet::new(),
            AgentConfig::default(),
        );
        let context = ctx();
        context.cancel.cancel();
        let out = agent.run(&context, "go").await.ok();
        assert_eq!(out.map(|answer| answer.status), Some(RunStatus::Cancelled));
    }

    #[tokio::test]
    async fn retryable_model_error_is_retried() {
        let (agent, sink) = agent(
            Scripted::new(vec![
                Err(ModelError::Transport("boom".into())),
                Ok(text("recovered")),
            ]),
            ToolSet::new(),
            AgentConfig::default(),
        );
        let out = agent.run(&ctx(), "go").await.ok();
        assert_eq!(out.map(|answer| answer.text), Some("recovered".into()));
        assert!(
            sink.records()
                .iter()
                .any(|record| matches!(record.event, TraceEvent::ModelError { retried: true, .. }))
        );
    }

    #[tokio::test]
    async fn usage_and_cost_accumulate() {
        let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
        let (agent, _) = agent(
            Scripted::new(vec![
                Ok(calls(vec![("c1", "echo", json!({}))])),
                Ok(text("ok")),
            ]),
            tools,
            AgentConfig {
                usd_per_m_prompt: 1.0,
                usd_per_m_completion: 2.0,
                ..AgentConfig::default()
            },
        );
        let out = agent.run(&ctx(), "go").await.ok();
        let out = out.as_ref();
        assert_eq!(out.map(|answer| answer.usage.prompt_tokens), Some(20));
        assert_eq!(out.map(|answer| answer.usage.completion_tokens), Some(10));
        assert!(out.is_some_and(|answer| (answer.cost_usd - 0.000_04).abs() < 1e-12));
    }

    #[test]
    fn secrets_are_redacted_from_traces() {
        let redacted = redact(&json!({"user": "a", "password": "b", "nested": {"api_key": "c"}}));
        assert_eq!(redacted["user"], "a");
        assert_eq!(redacted["password"], "[redacted]");
        assert_eq!(redacted["nested"]["api_key"], "[redacted]");
    }
}
