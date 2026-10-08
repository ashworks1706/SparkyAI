//! Product events and their export as spans.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};

use crate::analytics::{Analytics, SCOPE};
use crate::core::config::Analytics as Settings;
use crate::core::types::AnalyticsEvent;

#[test]
fn an_analytics_event_carries_its_asker_and_properties() {
    let event = AnalyticsEvent::new("discord_ask", &42_u64)
        .with("place", "thread")
        .with("latency_ms", 120);
    assert_eq!(event.event, "discord_ask");
    assert_eq!(event.distinct_id, "42");
    assert_eq!(event.properties["place"], "thread");
    assert_eq!(event.properties["latency_ms"], 120);
}

#[test]
fn analytics_stays_off_without_the_switch() {
    let off = Settings { enabled: false };
    assert!(!Analytics::start(&off).record(AnalyticsEvent::new("a", &1_u64)));
    assert!(Analytics::start(&Settings::default()).record(AnalyticsEvent::new("a", &1_u64)));
}

#[test]
fn a_product_event_is_exported_as_its_own_span() {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    opentelemetry::global::set_tracer_provider(provider.clone());
    let _ = provider.tracer(SCOPE);

    let recorded = Analytics::start(&Settings::default())
        .record(AnalyticsEvent::new("discord_ask", &42_u64).with("place", "thread"));
    assert!(recorded);
    let _ = provider.force_flush();
    let spans = exporter.get_finished_spans().unwrap_or_default();
    let ask = spans.iter().find(|s| s.name == "discord_ask");
    assert!(
        ask.is_some(),
        "{:?}",
        spans.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    let Some(ask) = ask else {
        return;
    };
    let attr = |key: &str| {
        ask.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.to_string())
    };
    assert_eq!(attr("user.id").as_deref(), Some("42"));
    assert_eq!(attr("place").as_deref(), Some("thread"));
    let _ = provider.shutdown();
}
