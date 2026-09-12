//! Span fields shared by the model calls of the loop and of a task.
//!
//! Each model span carries two attribute sets for the same values: gen_ai, which PostHog reads,
//! and OpenInference, which the Phoenix trace UI reads.

use tracing::Span;

use crate::agent::harness::safety::redact::{json, truncate};
use crate::core::types::model::ModelResponse;

/// Records the reply, the model that answered and the token counts on a model span.
pub(super) fn record_reply(span: &Span, response: &ModelResponse, limit: usize) {
    span.record("gen_ai.response.model", response.model.as_str());
    span.record("llm.model_name", response.model.as_str());
    span.record(
        "gen_ai.usage.input_tokens",
        i64::from(response.usage.prompt_tokens),
    );
    span.record(
        "gen_ai.usage.output_tokens",
        i64::from(response.usage.completion_tokens),
    );
    span.record(
        "llm.token_count.prompt",
        i64::from(response.usage.prompt_tokens),
    );
    span.record(
        "llm.token_count.completion",
        i64::from(response.usage.completion_tokens),
    );
    let shown = truncate(&json(&[response.as_message()]), limit);
    span.record("gen_ai.output.messages", shown.as_str());
    span.record("output.value", shown.as_str());
}
