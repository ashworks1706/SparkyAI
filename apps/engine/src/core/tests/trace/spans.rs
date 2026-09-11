//! The attributes PostHog reads from the model and agent spans.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use tracing_subscriber::layer::SubscriberExt;

use crate::agent::harness::tools::ToolSet;
use crate::core::tests::support::{Scripted, agent, ctx, text};
use crate::core::types::agent::AgentConfig;

fn attr(span: &SpanData, key: &str) -> Option<String> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| kv.value.to_string())
}

/// Runs one answered turn and returns the exported spans.
fn spans_of_one_turn() -> Vec<SpanData> {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
    let cfg = AgentConfig {
        provider_name: "llama.cpp".into(),
        model_name: "qwen".into(),
        ..AgentConfig::default()
    };
    let (agent, _) = agent(Scripted::new(vec![Ok(text("hello"))]), ToolSet::new(), cfg);
    let ctx = ctx();
    let Ok(rt) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return Vec::new();
    };
    let answered = tracing::subscriber::with_default(subscriber, || {
        rt.block_on(agent.run(&ctx, "what time is it"))
    });
    assert!(answered.is_ok(), "{answered:?}");
    let _ = provider.force_flush();
    exporter.get_finished_spans().unwrap_or_default()
}

#[test]
fn the_llm_span_carries_the_gen_ai_attributes() {
    let spans = spans_of_one_turn();
    let llm = spans.iter().find(|s| s.name == "llm");
    assert!(
        llm.is_some(),
        "{:?}",
        spans.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    let Some(llm) = llm else {
        return;
    };
    let ctx = ctx();
    for (key, want) in [
        ("gen_ai.operation.name", "chat"),
        ("gen_ai.provider.name", "llama.cpp"),
        ("gen_ai.request.model", "qwen"),
        ("gen_ai.response.model", "test"),
        ("gen_ai.usage.input_tokens", "10"),
        ("gen_ai.usage.output_tokens", "5"),
        ("sparky.span", "llm"),
        ("sparky.tools", "[]"),
        ("sparky.step", "1"),
        ("sparky.attempt", "0"),
    ] {
        assert_eq!(attr(llm, key).as_deref(), Some(want), "{key}");
    }
    assert!(attr(llm, "gen_ai.request.max_tokens").is_some());
    assert!(attr(llm, "gen_ai.request.temperature").is_some());
    let input = attr(llm, "gen_ai.input.messages").unwrap_or_default();
    assert!(
        input.starts_with('[') && input.contains("what time is it"),
        "{input}"
    );
    let output = attr(llm, "gen_ai.output.messages").unwrap_or_default();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap_or_default();
    assert_eq!(parsed.len(), 1, "{output}");
    assert!(output.contains("hello"), "{output}");
    assert_eq!(
        attr(llm, "posthog.distinct_id").as_deref(),
        Some(ctx.user_id.as_str())
    );
    assert!(attr(llm, "$ai_session_id").is_some());
}

#[test]
fn the_run_span_is_an_agent_invocation() {
    let spans = spans_of_one_turn();
    let run = spans.iter().find(|s| s.name == "agent.run");
    assert!(run.is_some());
    let Some(run) = run else {
        return;
    };
    assert_eq!(
        attr(run, "gen_ai.operation.name").as_deref(),
        Some("invoke_agent")
    );
    assert_eq!(attr(run, "gen_ai.agent.name").as_deref(), Some("sparky"));
    assert_eq!(
        attr(run, "sparky.input").as_deref(),
        Some("what time is it")
    );
    assert_eq!(attr(run, "sparky.output").as_deref(), Some("hello"));
    assert!(attr(run, "sparky.status").is_some());
    assert!(
        spans
            .iter()
            .all(|s| attr(s, "openinference.span.kind").is_none())
    );
}
