//! What one request loads before its first model call: history, memory, evidence. A public
//! request loads no memory unless the settings allow it.

use std::time::Instant;

use tracing::Instrument;
use tracing::field::Empty;

use super::{Agent, ms};
use crate::core::types::agent::AgentError;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::message::Message;
use crate::core::types::conversation::{Stored, Visibility};
use crate::core::types::knowledge::retrieval::RetrievalQuery;
use crate::core::types::memory::{Memory, MemoryQuery};
use crate::core::types::trace::TraceEvent;
use crate::runtime::harness::agent::run::Inputs;
use crate::runtime::harness::safety::redact::{json, truncate};

/// How many of the oldest turns do not fit budget, counted from the newest. The newest turn is
/// always kept.
fn overflowing(turns: &[Stored], budget: usize, chars_per_token: usize) -> usize {
    let mut spent = 0;
    let mut kept = 0;
    for stored in turns.iter().rev() {
        let cost = stored.message.estimated_tokens(chars_per_token);
        if spent + cost > budget {
            break;
        }
        spent += cost;
        kept += 1;
    }
    turns.len().saturating_sub(kept.max(1))
}

impl Agent {
    /// The stored history of the conversation, compacted when it overflows its budget.
    pub(super) async fn history(&self, ctx: &RequestContext) -> Result<Vec<Message>, AgentError> {
        let Some(store) = &self.deps.conversations else {
            return Ok(Vec::new());
        };
        let loaded = store
            .load(ctx, self.cfg.history_turns)
            .await
            .map_err(|error| AgentError::Store(error.to_string()))?;
        Ok(self.compacted(ctx, loaded).await)
    }

    /// Replaces the turns that do not fit the history budget with one stored summary that
    /// covers them, and returns the history the prompt carries.
    ///
    /// A failed compaction returns the turns unchanged.
    async fn compacted(&self, ctx: &RequestContext, turns: Vec<Stored>) -> Vec<Message> {
        let Some(compactor) = &self.deps.compactor else {
            return turns.into_iter().map(|s| s.message).collect();
        };
        let cpt = self.cfg.budget.chars_per_token;
        let overflow = overflowing(&turns, self.cfg.budget.history, cpt);
        let Some(covers) = overflow.checked_sub(1).map(|last| turns[last].position) else {
            return turns.into_iter().map(|s| s.message).collect();
        };
        let replaced: Vec<Message> = turns[..overflow]
            .iter()
            .map(|s| s.message.clone())
            .collect();
        self.deps.trace.emit(
            ctx,
            TraceEvent::Compaction {
                turns: replaced.len(),
            },
        );
        let summary = match compactor.compact(ctx, &replaced).await {
            Ok(summary) => summary,
            Err(error) => {
                tracing::warn!(error = %error, "compaction failed; history is trimmed instead");
                return turns.into_iter().map(|s| s.message).collect();
            }
        };
        if let Some(store) = &self.deps.conversations
            && let Err(error) = store.append_summary(ctx, &summary, covers).await
        {
            // The prompt still gets the summary.
            tracing::warn!(error = %error, "compacted turn was not stored");
        }
        let mut out = Vec::with_capacity(turns.len() - overflow + 1);
        out.push(summary);
        out.extend(turns.into_iter().skip(overflow).map(|s| s.message));
        out
    }

    /// Loads the history, memory, and evidence of one request.
    pub(super) async fn load(
        &self,
        ctx: &RequestContext,
        input: &str,
    ) -> Result<Inputs, AgentError> {
        let deps = &self.deps;
        let history = self.history(ctx).await?;
        let memory = if ctx.visibility == Visibility::Public && !self.cfg.recall_in_public {
            Vec::new()
        } else {
            let recalled = match &deps.memory {
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
            self.with_profile(ctx, recalled).await
        };
        deps.trace.emit(
            ctx,
            TraceEvent::MemoryRecalled {
                count: memory.len(),
            },
        );
        let evidence = match &deps.retriever {
            Some(retriever) => {
                let started = Instant::now();
                let query = RetrievalQuery::new(input, self.cfg.retrieval_top_k);
                let asked = truncate(input, self.cfg.max_span_value_chars);
                let span = tracing::info_span!(
                    "retrieve",
                    "gen_ai.operation.name" = "retrieval",
                    "sparky.input" = %asked,
                    "sparky.output" = Empty,
                    // OpenInference, read by the Phoenix trace UI.
                    "openinference.span.kind" = "RETRIEVER",
                    "input.value" = %asked,
                    "output.value" = Empty,
                    "session.id" = %ctx.conversation_id,
                    "user.id" = %ctx.user_id,
                    "$ai_session_id" = %ctx.conversation_id,
                    "posthog.distinct_id" = %ctx.user_id,
                    "otel.status_code" = Empty,
                );
                let found = retriever
                    .retrieve(ctx, &query)
                    .instrument(span.clone())
                    .await
                    .map_err(|error| AgentError::Store(format!("retrieval: {error}")))?;
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
                let shown = truncate(&json(&listing), self.cfg.max_span_value_chars);
                span.record("sparky.output", shown.as_str());
                span.record("output.value", shown.as_str());
                span.record("otel.status_code", "OK");
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
            None => Vec::new(),
        };
        Ok(Inputs {
            history,
            memory,
            evidence,
        })
    }

    /// Appends the profile nodes and relations of this user to the recalled memories. A failed
    /// graph read is logged and adds nothing.
    async fn with_profile(&self, ctx: &RequestContext, mut memory: Vec<Memory>) -> Vec<Memory> {
        let Some(graph) = &self.deps.profile_graph else {
            return memory;
        };
        let limit = self.cfg.memory_recall_limit;
        match graph.recall(ctx, limit).await {
            Ok(nodes) => memory.extend(nodes.iter().map(Memory::from)),
            Err(error) => tracing::warn!(error = %error, "profile graph recall failed"),
        }
        match graph.relations(ctx, limit).await {
            Ok(relations) => memory.extend(relations.iter().map(Memory::from)),
            Err(error) => tracing::warn!(error = %error, "profile relation recall failed"),
        }
        memory
    }
}
