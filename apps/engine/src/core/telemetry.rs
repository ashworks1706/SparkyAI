//! Logging and OpenTelemetry export over OTLP/HTTP protobuf to Phoenix.

use std::collections::HashMap;
use std::time::Duration;

use opentelemetry::{KeyValue, trace::TracerProvider as _};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{Resource, trace::Sampler, trace::SdkTracerProvider};
use secrecy::ExposeSecret;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

use crate::core::config::Telemetry;

/// Keeps the OTLP exporter alive; flushes on drop. Drop it outside a tokio runtime.
pub struct Guard {
    otel: Option<SdkTracerProvider>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(p) = self.otel.take()
            && let Err(error) = p.shutdown()
        {
            tracing::warn!(%error, "trace export shutdown failed");
        }
    }
}

/// The OTLP/HTTP traces path, fixed by the protocol. Phoenix serves it under phoenix_url.
const OTLP_TRACES_PATH: &str = "/v1/traces";

/// Resource attribute naming the Phoenix project a span belongs to.
const PROJECT_NAME: &str = "openinference.project.name";

/// The Phoenix traces endpoint, or None when phoenix_url is unset or empty.
pub fn phoenix_url(cfg: &Telemetry) -> Option<String> {
    cfg.phoenix_url
        .as_deref()
        .map(|url| url.trim().trim_end_matches('/'))
        .filter(|url| !url.is_empty())
        .map(|url| format!("{url}{OTLP_TRACES_PATH}"))
}

/// Why export is off, or None when Phoenix is configured.
fn disabled(cfg: &Telemetry) -> Option<&'static str> {
    phoenix_url(cfg)
        .is_none()
        .then_some("telemetry.phoenix_url is unset; trace export is off")
}

/// The bearer header, empty when phoenix_api_key is empty.
fn headers(cfg: &Telemetry) -> HashMap<String, String> {
    let key = cfg.phoenix_api_key.expose_secret().trim();
    if key.is_empty() {
        return HashMap::new();
    }
    HashMap::from([("Authorization".to_owned(), format!("Bearer {key}"))])
}

/// One OTLP/HTTP protobuf exporter to url.
fn exporter(cfg: &Telemetry, url: String) -> anyhow::Result<SpanExporter> {
    Ok(SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(url)
        .with_headers(headers(cfg))
        .with_timeout(Duration::from_secs(cfg.export_timeout_secs))
        .build()?)
}

/// The resource every exported span carries: service, environment, and Phoenix project.
pub fn resource(cfg: &Telemetry, service: &str, env: &str) -> Resource {
    Resource::builder()
        .with_service_name(service.to_owned())
        .with_attribute(KeyValue::new("deployment.environment", env.to_owned()))
        .with_attribute(KeyValue::new(PROJECT_NAME, cfg.project_name.clone()))
        .build()
}

/// The tracer provider exporting every span to Phoenix, or None when it is not configured.
pub fn provider(
    cfg: &Telemetry,
    service: &str,
    env: &str,
) -> anyhow::Result<Option<SdkTracerProvider>> {
    let Some(url) = phoenix_url(cfg) else {
        return Ok(None);
    };
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter(cfg, url)?)
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
            cfg.sample_ratio,
        ))))
        .with_resource(resource(cfg, service, env))
        .build();
    Ok(Some(provider))
}

/// Installs the global tracing subscriber with fmt and an optional OTLP layer; call outside tokio.
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

    let filter = match std::env::var("RUST_LOG") {
        Ok(spec) => {
            EnvFilter::try_new(&spec).map_err(|e| anyhow::anyhow!("RUST_LOG {spec:?}: {e}"))?
        }
        Err(_) => EnvFilter::try_new(log_level)
            .map_err(|e| anyhow::anyhow!("app.log_level {log_level:?}: {e}"))?,
    };
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
