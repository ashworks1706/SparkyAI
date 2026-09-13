//! One step of the loop: assemble the prompt, call the model, check the response, and run what
//! it asked for.

use super::{Agent, StepOutcome};
use crate::core::types::agent::assemble::Sections;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::message::Message;
use crate::core::types::model::{FinishReason, ModelError, ModelResponse};
use crate::core::types::safety::guardrail::{Stage, Verdict};
use crate::core::types::trace::{RunStatus, TraceEvent};
use crate::runtime::harness::agent::execute::HeldError;
use crate::runtime::harness::agent::prompt::assemble;
use crate::runtime::harness::agent::run::{Inputs, Run};
use crate::runtime::harness::safety::redact::truncate;

/// The outcome of a step that asked for no capabilities.
fn answered(response: &ModelResponse, force_answer: bool) -> StepOutcome {
    let written_call = force_answer && is_call_text(&response.content);
    if !response.content.trim().is_empty() && !written_call {
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

/// Whether text is a tool call written out as JSON rather than an answer.
fn is_call_text(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text.trim())
        .is_ok_and(|v| v.get("name").is_some() && v.get("arguments").is_some())
}

impl Agent {
    /// One model call and whatever tool calls it asks for.
    pub(super) async fn step(
        &self,
        run: &mut Run<'_>,
        inputs: &Inputs,
    ) -> Result<StepOutcome, ModelError> {
        let ctx = run.ctx;
        let messages = self.prompt_for(run, inputs);
        let thinking = self.thinking(run, inputs);
        self.deps
            .trace
            .emit(ctx, TraceEvent::ModelStarted { step: run.steps });
        let (response, thought) = self.answer_with_thinking(run, &messages, thinking).await?;
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
            // A step that showed a thought keeps it in place of the answered line.
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

    /// Assembles the prompt of the next model call within the budget left once the tool schemas
    /// are counted, and records what fit.
    fn prompt_for(&self, run: &mut Run<'_>, inputs: &Inputs) -> Vec<Message> {
        let ctx = run.ctx;
        // The turns of this request after the input, which assembly appends itself.
        let turn = run
            .new_turns
            .get(run.appended_by_assembly..)
            .unwrap_or_default();
        let cpt = self.cfg.budget.chars_per_token;
        // Tool schemas ride along with every request and come out of the same budget.
        let tool_tokens = self.deps.tools.estimated_tokens(cpt);
        let mut budget = self.cfg.budget;
        budget.total = budget.total.saturating_sub(tool_tokens);
        let assembled = assemble::assemble(
            ctx,
            &Sections {
                system: &self.system_prompt,
                memory: &inputs.memory,
                evidence: &inputs.evidence,
                history: &inputs.history,
                turn,
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
        let mut messages = assembled.messages;
        if run.force_answer {
            messages.push(Message::system(self.prompt.answer_only_line.clone()));
        }
        messages
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
        // An empty capability branch is not checked.
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
}
