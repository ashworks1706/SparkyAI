//! Product events: one span each, exported to Phoenix through the telemetry provider.

use opentelemetry::trace::{Span as _, Tracer as _};
use opentelemetry::{KeyValue, Value, global};
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

use crate::core::config::Analytics as Settings;
use crate::core::types::AnalyticsEvent;

/// Instrumentation scope the event spans are recorded under.
pub const SCOPE: &str = "discord.analytics";

/// Handle that records product events. Cheap to clone; a disabled handle drops everything.
#[derive(Debug, Clone)]
pub struct Analytics {
    enabled: bool,
}

impl Analytics {
    /// A handle that follows analytics.enabled.
    pub fn start(settings: &Settings) -> Self {
        if !settings.enabled {
            tracing::info!("analytics disabled by analytics.enabled");
        }
        Self {
            enabled: settings.enabled,
        }
    }

    /// Records event as one span under the interaction. Returns false when it was dropped.
    pub fn record(&self, event: AnalyticsEvent) -> bool {
        if !self.enabled {
            return false;
        }
        let name = event.event;
        let tracer = global::tracer(SCOPE);
        let mut span = tracer
            .span_builder(name)
            .with_attributes(attributes(event))
            .start_with_context(&tracer, &tracing::Span::current().context());
        span.end();
        true
    }
}

/// The span attributes of one event: the asker, then every property.
fn attributes(event: AnalyticsEvent) -> Vec<KeyValue> {
    let mut attrs = vec![KeyValue::new("user.id", event.distinct_id)];
    attrs.extend(
        event
            .properties
            .into_iter()
            .map(|(key, value)| KeyValue::new(key, attribute_value(value))),
    );
    attrs
}

/// A property as an OTel value. Anything that is not a scalar is kept as its JSON.
fn attribute_value(value: serde_json::Value) -> Value {
    match value {
        serde_json::Value::Bool(v) => Value::Bool(v),
        serde_json::Value::String(v) => Value::String(v.into()),
        serde_json::Value::Number(ref v) => v
            .as_i64()
            .map(Value::I64)
            .or_else(|| v.as_f64().map(Value::F64))
            .unwrap_or_else(|| Value::String(v.to_string().into())),
        other => Value::String(other.to_string().into()),
    }
}
