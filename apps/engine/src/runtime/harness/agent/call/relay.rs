//! A streaming model call relayed live: reasoning as thinking, answer as draft, guardrail checked.

use tokio::sync::mpsc::unbounded_channel;

use super::draft::{Draft, Release};
use crate::core::types::agent::context::RequestContext;
use crate::core::types::model::{ModelError, ModelRequest, ModelResponse};
use crate::core::types::safety::guardrail::{Stage, Verdict};
use crate::core::types::trace::TraceEvent;
use crate::runtime::harness::agent::Agent;
use crate::runtime::harness::safety::redact::truncate;

impl Agent {
    /// Runs one completion. When streaming, output shows live; a draft not kept withdraws.
    pub(super) async fn relay(
        &self,
        ctx: &RequestContext,
        step: u32,
        request: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        if !self.cfg.stream {
            return self.deps.model.generate(ctx, request).await;
        }
        let (tx, mut rx) = unbounded_channel();
        let mut draft = Draft::new(self.cfg.stream_block_chars);
        let call = self.deps.model.stream(ctx, request, tx);
        tokio::pin!(call);
        let result = loop {
            tokio::select! {
                biased;
                Some(delta) = rx.recv() => {
                    for release in draft.push(delta) {
                        self.show(ctx, step, &mut draft, release).await;
                    }
                }
                result = &mut call => break result,
            }
        };
        while let Ok(delta) = rx.try_recv() {
            for release in draft.push(delta) {
                self.show(ctx, step, &mut draft, release).await;
            }
        }
        let withdrawn = match &result {
            Ok(response) => !response.tool_calls.is_empty(),
            Err(_) => true,
        };
        if withdrawn && draft.shown() {
            self.deps
                .trace
                .emit(ctx, TraceEvent::AnswerDraftCleared { step });
        }
        result
    }

    /// Shows one release. A refused answer block withdraws the draft and ends this call's showing.
    async fn show(&self, ctx: &RequestContext, step: u32, draft: &mut Draft, release: Release) {
        let limit = self.cfg.max_span_value_chars;
        let event = match release {
            Release::Reasoning(text) => TraceEvent::ModelReasoning {
                step,
                text: truncate(&text, limit),
            },
            Release::Thought(text) => TraceEvent::ModelThought {
                step,
                text: truncate(&text, limit),
            },
            Release::Answer(text) => {
                if let Some(guardrail) = &self.deps.guardrail
                    && let Verdict::Block { .. } = guardrail.check(ctx, Stage::Answer, &text).await
                {
                    let shown = draft.shown();
                    draft.withhold();
                    if shown {
                        self.deps
                            .trace
                            .emit(ctx, TraceEvent::AnswerDraftCleared { step });
                    }
                    return;
                }
                TraceEvent::AnswerDraft {
                    step,
                    text: truncate(&text, limit),
                }
            }
        };
        self.deps.trace.emit(ctx, event);
    }
}
