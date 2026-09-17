//! Logging and OTLP/HTTP export to Phoenix; one span per interaction, shares session.

use std::collections::HashMap;
use std::time::Duration;

use opentelemetry::{KeyValue, trace::TracerProvider as _};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{Resource, trace::Sampler, trace::SdkTracerProvider};
use secrecy::ExposeSecret;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

use crate::core::config::Telemetry;

/// Default service.name and span-target prefix for this binary.
const SERVICE: &str = "discord";

/// The OTLP/HTTP traces path, fixed by the protocol. Phoenix serves it under phoenix_url.
const OTLP_TRACES_PATH: &str = "/v1/traces";

/// Keeps the OTLP exporter alive. Flushes on drop.
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
            if let Err(e) = provider.shutdown() {
                tracing::warn!(error = %e, "telemetry shutdown failed");
            }
        })
        .join();
        if done.is_err() {
            tracing::warn!("telemetry shutdown thread failed");
        }
    }
}

/// Resource attribute naming the Phoenix project a span belongs to.
const PROJECT_NAME: &str = "openinference.project.name";

/// The Phoenix traces endpoint, or None when phoenix_url is empty.
pub fn phoenix_target(cfg: &Telemetry) -> Option<String> {
    cfg.phoenix_url
        .as_deref()
        .map(|url| url.trim().trim_end_matches('/'))
        .filter(|url| !url.is_empty())
        .map(|url| format!("{url}{OTLP_TRACES_PATH}"))
}

/// The resource every exported span carries: service, environment, and Phoenix project.
pub fn resource(cfg: &Telemetry, service: &str, env: &str) -> Resource {
    Resource::builder()
        .with_service_name(service.to_owned())
        .with_attribute(KeyValue::new("deployment.environment", env.to_owned()))
        .with_attribute(KeyValue::new(PROJECT_NAME, cfg.project_name.clone()))
        .build()
}

/// One tracer provider exporting every span to Phoenix; None when it is not configured.
pub fn provider(
    cfg: &Telemetry,
    service: &str,
    env: &str,
) -> anyhow::Result<Option<SdkTracerProvider>> {
    let Some(url) = phoenix_target(cfg) else {
        return Ok(None);
    };
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter(
            url,
            cfg.phoenix_api_key.expose_secret().trim(),
            Duration::from_secs(cfg.export_timeout_secs),
        )?)
        // Samples whole traces by trace id; children follow the root decision.
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
            cfg.sample_ratio,
        ))))
        .with_resource(resource(cfg, service, env))
        .build();
    Ok(Some(provider))
}

/// An OTLP/HTTP span exporter to url, with a bearer token when key is not empty.
fn exporter(url: String, key: &str, timeout: Duration) -> anyhow::Result<SpanExporter> {
    let headers = if key.is_empty() {
        HashMap::new()
    } else {
        HashMap::from([("Authorization".to_owned(), format!("Bearer {key}"))])
    };
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
        tracing::warn!("telemetry.phoenix_url is unset; span export is off");
    }
    Ok(Guard::new(otel))
}
