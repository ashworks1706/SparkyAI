//! Logging and OpenTelemetry export over OTLP/HTTP protobuf to PostHog and Phoenix.
//! PostHog receives every span with the project token as bearer; Phoenix receives them without
//! authentication. An empty host or token turns PostHog off, an empty phoenix_url turns Phoenix
//! off, and export runs while either is set.

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
        if let Some(p) = self.otel.take()
            && let Err(error) = p.shutdown()
        {
            tracing::warn!(%error, "trace export shutdown failed");
        }
    }
}

/// The OTLP/HTTP traces path, fixed by the protocol. Phoenix serves it under phoenix_url.
const OTLP_TRACES_PATH: &str = "/v1/traces";

/// A base URL without a trailing slash, or None when it is unset or empty.
fn base(url: Option<&str>) -> Option<&str> {
    url.map(|h| h.trim().trim_end_matches('/'))
        .filter(|h| !h.is_empty())
}

/// The PostHog endpoints, empty when the host or the token is empty. The AI endpoint is
/// included only when ai_path is set.
fn posthog_urls(cfg: &Telemetry) -> Vec<String> {
    let Some(host) = base(cfg.host.as_deref()) else {
        return Vec::new();
    };
    if cfg.project_token.expose_secret().trim().is_empty() {
        return Vec::new();
    }
    let mut urls = vec![format!("{host}{}", cfg.traces_path)];
    if !cfg.ai_path.trim().is_empty() {
        urls.push(format!("{host}{}", cfg.ai_path));
    }
    urls
}

/// The Phoenix endpoint, or None when phoenix_url is empty.
fn phoenix_url(cfg: &Telemetry) -> Option<String> {
    base(cfg.phoenix_url.as_deref()).map(|url| format!("{url}{OTLP_TRACES_PATH}"))
}

/// Why export is off, or None when at least one destination is configured.
fn disabled(cfg: &Telemetry) -> Option<&'static str> {
    (posthog_urls(cfg).is_empty() && phoenix_url(cfg).is_none())
        .then_some("telemetry.host, telemetry.project_token and telemetry.phoenix_url are unset; trace export is off")
}

/// One OTLP/HTTP protobuf exporter to url, with the given headers.
fn exporter(
    cfg: &Telemetry,
    url: String,
    headers: HashMap<String, String>,
) -> anyhow::Result<SpanExporter> {
    Ok(SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(url)
        .with_headers(headers)
        .with_timeout(Duration::from_secs(cfg.export_timeout_secs))
        .build()?)
}

/// The bearer header PostHog authenticates with.
fn bearer(cfg: &Telemetry) -> HashMap<String, String> {
    HashMap::from([(
        "Authorization".to_owned(),
        format!("Bearer {}", cfg.project_token.expose_secret().trim()),
    )])
}

/// The tracer provider exporting every span to each configured destination, or None when none
/// is configured. Build it and shut it down outside a tokio runtime.
pub fn provider(
    cfg: &Telemetry,
    service: &str,
    env: &str,
) -> anyhow::Result<Option<SdkTracerProvider>> {
    if disabled(cfg).is_some() {
        return Ok(None);
    }
    let mut builder = SdkTracerProvider::builder();
    for url in posthog_urls(cfg) {
        builder = builder.with_batch_exporter(exporter(cfg, url, bearer(cfg))?);
    }
    if let Some(url) = phoenix_url(cfg) {
        builder = builder.with_batch_exporter(exporter(cfg, url, HashMap::new())?);
    }
    let provider = builder
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
