//! Logging and `OpenTelemetry` export over OTLP/gRPC to Phoenix (or any OTLP collector).
//! Spans carry `OpenInference` attributes. An empty endpoint disables export.

use std::time::Duration;

use opentelemetry::{KeyValue, trace::TracerProvider as _};
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::{Resource, trace::Sampler, trace::SdkTracerProvider};
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

use crate::core::config::Telemetry;

/// Keeps the OTLP exporter alive; flushes on drop.
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

/// Installs the global `tracing` subscriber with fmt and optional OTLP layers.
pub fn init(cfg: &Telemetry, service: &str, env: &str, log_level: &str) -> anyhow::Result<Guard> {
    let service = cfg
        .service_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(service)
        .to_owned();
    let service = service.as_str();
    let otel = match cfg
        .otlp_endpoint
        .as_deref()
        .filter(|e| !e.trim().is_empty())
    {
        Some(endpoint) => {
            let exporter = SpanExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint)
                .with_timeout(Duration::from_secs(cfg.export_timeout_secs))
                .build()?;
            let provider = SdkTracerProvider::builder()
                .with_batch_exporter(exporter)
                // Ratio sampling keeps whole traces: a sampled root carries its children, so a
                // ratio below 1 thins traces rather than punching holes in them.
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
            opentelemetry::global::set_tracer_provider(provider.clone());
            Some(provider)
        }
        None => None,
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    let fmt = if env == "development" {
        tracing_subscriber::fmt::layer().pretty().boxed()
    } else {
        tracing_subscriber::fmt::layer().json().boxed()
    };
    // Export only this crate's spans. Dependencies (serenity's gateway, Rig, tower-http)
    // instrument themselves too, and that noise would bury the request tree in Phoenix.
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

    Ok(Guard { otel })
}
