//! Logging and OpenTelemetry export over OTLP/HTTP protobuf to PostHog, to the traces path
//! and the AI path. One span per interaction, sharing the engine session id.

use std::collections::HashMap;
use std::time::Duration;

use opentelemetry::{KeyValue, trace::TracerProvider as _};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{Resource, trace::Sampler, trace::SdkTracerProvider};
use secrecy::{ExposeSecret, SecretString};
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

use crate::core::config::Telemetry;

/// Default service.name and span-target prefix for this binary.
const SERVICE: &str = "discord";

/// Keeps the OTLP exporters alive. Flushes on drop.
pub struct Guard {
    otel: Option<SdkTracerProvider>,
}

impl Guard {
    /// Holds provider until the guard drops.
    pub fn new(otel: Option<SdkTracerProvider>) -> Self {
        Self { otel }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let Some(provider) = self.otel.take() else {
            return;
        };
        // The blocking HTTP client must not be dropped on a tokio thread.
        let done = std::thread::spawn(move || {
            let _ = provider.shutdown();
        })
        .join();
        if done.is_err() {
            tracing::warn!("telemetry shutdown thread failed");
        }
    }
}

/// The PostHog base URL and project token, when both are set.
pub fn export_target(cfg: &Telemetry) -> Option<(&str, &SecretString)> {
    let host = cfg
        .host
        .as_deref()
        .map(str::trim)
        .filter(|h| !h.is_empty())?;
    let token = &cfg.project_token;
    (!token.expose_secret().trim().is_empty()).then_some((host.trim_end_matches('/'), token))
}

/// One tracer provider exporting every span to host plus traces_path and host plus ai_path.
/// None when export is disabled.
pub fn provider(
    cfg: &Telemetry,
    service: &str,
    env: &str,
) -> anyhow::Result<Option<SdkTracerProvider>> {
    let Some((host, token)) = export_target(cfg) else {
        return Ok(None);
    };
    let timeout = Duration::from_secs(cfg.export_timeout_secs);
    let traces = exporter(format!("{host}{}", cfg.traces_path), token, timeout)?;
    let ai = exporter(format!("{host}{}", cfg.ai_path), token, timeout)?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(traces)
        .with_batch_exporter(ai)
        // Ratio sampling keeps whole traces. A sampled root carries its children.
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
            cfg.sample_ratio,
        ))))
        .with_resource(
            Resource::builder()
                .with_service_name(service.to_owned())
                .with_attribute(KeyValue::new("deployment.environment", env.to_owned()))
                .build(),
        )
        .build();
    Ok(Some(provider))
}

/// A protobuf OTLP/HTTP span exporter to url with the bearer token.
fn exporter(url: String, token: &SecretString, timeout: Duration) -> anyhow::Result<SpanExporter> {
    let headers = HashMap::from([(
        "Authorization".to_owned(),
        format!("Bearer {}", token.expose_secret()),
    )]);
    Ok(SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(url)
        .with_timeout(timeout)
        .with_headers(headers)
        .build()?)
}

/// Installs the global tracing subscriber with fmt and optional OTLP layers.
pub fn init(cfg: &Telemetry, env: &str, log_level: &str) -> anyhow::Result<Guard> {
    let service = cfg
        .service_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(SERVICE)
        .to_owned();
    let otel = provider(cfg, &service, env)?;
    if let Some(p) = &otel {
        opentelemetry::global::set_tracer_provider(p.clone());
    }
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    let fmt = if env == "development" {
        tracing_subscriber::fmt::layer().pretty().boxed()
    } else {
        tracing_subscriber::fmt::layer().json().boxed()
    };
    // Export only the spans of this crate.
    let prefix = cfg
        .span_target_prefix
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .unwrap_or(SERVICE)
        .to_owned();
    let own_spans = filter_fn(move |meta| meta.target().starts_with(&prefix));
    let otel_layer = otel.as_ref().map(|p| {
        tracing_opentelemetry::layer()
            .with_tracer(p.tracer(service.clone()))
            .with_filter(own_spans)
    });
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt)
        .with(otel_layer)
        .init();
    if otel.is_none() {
        tracing::warn!("telemetry.host or telemetry.project_token is empty; span export is off");
    }
    Ok(Guard::new(otel))
}
