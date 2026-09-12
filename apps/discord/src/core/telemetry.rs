//! Logging and OpenTelemetry export over OTLP/HTTP protobuf to PostHog, to the traces path
//! and the AI path, and to Phoenix. Each destination is independent. One span per interaction,
//! sharing the engine session id.

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

/// The OTLP/HTTP traces path, fixed by the protocol. Phoenix serves it under phoenix_url.
const OTLP_TRACES_PATH: &str = "/v1/traces";

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

/// The Phoenix traces endpoint, or None when phoenix_url is empty.
pub fn phoenix_target(cfg: &Telemetry) -> Option<String> {
    cfg.phoenix_url
        .as_deref()
        .map(|url| url.trim().trim_end_matches('/'))
        .filter(|url| !url.is_empty())
        .map(|url| format!("{url}{OTLP_TRACES_PATH}"))
}

/// One tracer provider exporting every span to each configured destination: host plus
/// traces_path, host plus ai_path, and Phoenix. None when no destination is configured.
pub fn provider(
    cfg: &Telemetry,
    service: &str,
    env: &str,
) -> anyhow::Result<Option<SdkTracerProvider>> {
    let posthog = export_target(cfg);
    let phoenix = phoenix_target(cfg);
    if posthog.is_none() && phoenix.is_none() {
        return Ok(None);
    }
    let timeout = Duration::from_secs(cfg.export_timeout_secs);
    let mut builder = SdkTracerProvider::builder();
    if let Some((host, token)) = posthog {
        builder = builder.with_batch_exporter(exporter(
            format!("{host}{}", cfg.traces_path),
            Some(token),
            timeout,
        )?);
        // The self-hosted capture-ai service takes PostHog's own event payload, not OTLP, so
        // the AI endpoint is off unless ai_path says otherwise.
        if !cfg.ai_path.trim().is_empty() {
            builder = builder.with_batch_exporter(exporter(
                format!("{host}{}", cfg.ai_path),
                Some(token),
                timeout,
            )?);
        }
    }
    if let Some(url) = phoenix {
        builder = builder.with_batch_exporter(exporter(url, None, timeout)?);
    }
    let provider = builder
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

/// A protobuf OTLP/HTTP span exporter to url, with the bearer token when the destination
/// authenticates.
fn exporter(
    url: String,
    token: Option<&SecretString>,
    timeout: Duration,
) -> anyhow::Result<SpanExporter> {
    let headers = token.map_or_else(HashMap::new, |token| {
        HashMap::from([(
            "Authorization".to_owned(),
            format!("Bearer {}", token.expose_secret()),
        )])
    });
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
        tracing::warn!(
            "telemetry.host, telemetry.project_token and telemetry.phoenix_url are unset; span export is off"
        );
    }
    Ok(Guard::new(otel))
}
