//! What one request loads before its first model call: history, memory, evidence. A public
//! request loads no memory unless the settings allow it.

use std::time::Instant;

use tracing::Instrument;
use tracing::field::Empty;

use super::{Agent, ms};
use crate::agent::harness::agent::run::Inputs;
use crate::agent::harness::safety::redact::{json, truncate};
use crate::core::types::agent::AgentError;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::conversation::message::Message;
use crate::core::types::knowledge::retrieval::RetrievalQuery;
use crate::core::types::memory::{Memory, MemoryQuery};
use crate::core::types::trace::TraceEvent;

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

impl Agent {
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
    /// Appends what the graph knows about this user to the memories recalled. A graph that
    /// cannot be read leaves the prompt with the memories alone.
    async fn with_profile(&self, ctx: &RequestContext, mut memory: Vec<Memory>) -> Vec<Memory> {
        let Some(graph) = &self.deps.profile_graph else {
            return memory;
        };
        let limit = self.cfg.memory_recall_limit;
        match graph.recall(ctx, limit).await {
            Ok(nodes) => memory.extend(nodes.iter().map(Memory::from)),
            Err(error) => tracing::warn!(error = %error, "profile graph recall failed"),
        }
        // A node says what the user is connected to. A relation says how.
        match graph.relations(ctx, limit).await {
            Ok(relations) => memory.extend(relations.iter().map(Memory::from)),
            Err(error) => tracing::warn!(error = %error, "profile relation recall failed"),
        }
        memory
    }
}
