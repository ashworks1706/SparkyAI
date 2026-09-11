//! Logging and OpenTelemetry export over OTLP/HTTP protobuf to PostHog.
//! Every span goes to the traces path and the AI path of the telemetry host, authenticated with
//! the project token. An empty host or token disables export.

use std::collections::HashMap;
use std::time::Duration;

use opentelemetry::{KeyValue, trace::TracerProvider as _};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{Resource, trace::Sampler, trace::SdkTracerProvider};
use secrecy::ExposeSecret;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

use crate::core::config::Telemetry;

/// Keeps the OTLP exporters alive; flushes on drop. Drop it outside a tokio runtime.
pub struct Guard {
    otel: Option<SdkTracerProvider>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(p) = self.otel.take() {
            let _ = p.shutdown();
        }
    }
}

/// The host without a trailing slash, or None when it is unset or empty.
fn host(cfg: &Telemetry) -> Option<&str> {
    cfg.host
        .as_deref()
        .map(|h| h.trim().trim_end_matches('/'))
        .filter(|h| !h.is_empty())
}

/// Why export is off, or None when the host and the token are both set.
fn disabled(cfg: &Telemetry) -> Option<&'static str> {
    if host(cfg).is_none() {
        return Some("telemetry.host is empty; trace export is off");
    }
    if cfg.project_token.expose_secret().trim().is_empty() {
        return Some("telemetry.project_token is empty; trace export is off");
    }
    None
}

/// One OTLP/HTTP protobuf exporter to url, carrying the project token as a bearer token.
fn exporter(cfg: &Telemetry, url: String) -> anyhow::Result<SpanExporter> {
    let headers = HashMap::from([(
        "Authorization".to_owned(),
        format!("Bearer {}", cfg.project_token.expose_secret().trim()),
    )]);
    Ok(SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(url)
        .with_headers(headers)
        .with_timeout(Duration::from_secs(cfg.export_timeout_secs))
        .build()?)
}

/// The tracer provider exporting every span to host plus traces_path and host plus ai_path,
/// or None when export is off. Build it and shut it down outside a tokio runtime.
pub fn provider(
    cfg: &Telemetry,
    service: &str,
    env: &str,
) -> anyhow::Result<Option<SdkTracerProvider>> {
    let Some(host) = host(cfg).filter(|_| disabled(cfg).is_none()) else {
        return Ok(None);
    };
    let traces = exporter(cfg, format!("{host}{}", cfg.traces_path))?;
    let ai = exporter(cfg, format!("{host}{}", cfg.ai_path))?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(traces)
        .with_batch_exporter(ai)
        // Ratio sampling keeps whole traces: a sampled root carries its children.
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

/// Installs the global tracing subscriber with fmt and optional OTLP layers. Call it outside a
/// tokio runtime.
pub fn init(cfg: &Telemetry, service: &str, env: &str, log_level: &str) -> anyhow::Result<Guard> {
    let service = cfg
        .service_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(service)
        .to_owned();
    let service = service.as_str();
    let otel = provider(cfg, service, env)?;
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
        .unwrap_or("engine")
        .to_owned();
    let own_spans = filter_fn(move |meta| meta.target().starts_with(&prefix));
    let otel_layer = otel.as_ref().map(|p| {
        tracing_opentelemetry::layer()
            .with_tracer(p.tracer(service.to_owned()))
            .with_filter(own_spans)
    });

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt)
        .with(otel_layer)
        .init();
    if let Some(reason) = disabled(cfg) {
        tracing::warn!("{reason}");
    }

    Ok(Guard { otel })
}
