//! Span fields shared by loop and task model calls: gen_ai for PostHog, OpenInference for Phoenix.

use tracing::Span;

use crate::core::types::model::ModelResponse;
use crate::runtime::harness::safety::redact::{json, truncate};

/// Records the reply, the model that answered and the token counts on a model span.
pub(in crate::runtime::harness::agent) fn record_reply(
    span: &Span,
    response: &ModelResponse,
    limit: usize,
) {
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
