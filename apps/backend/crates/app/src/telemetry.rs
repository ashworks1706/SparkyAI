//! Logging, Sentry, and OpenTelemetry export (Axiom via OTLP).

use crate::config::Telemetry;
use opentelemetry::{KeyValue, trace::TracerProvider as _};
use opentelemetry_otlp::{SpanExporter, WithExportConfig, WithTonicConfig};
use opentelemetry_sdk::{Resource, trace::SdkTracerProvider};
use secrecy::ExposeSecret;
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

pub struct Guard {
    _sentry: Option<sentry::ClientInitGuard>,
    otel: Option<SdkTracerProvider>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(p) = self.otel.take() {
            let _ = p.shutdown();
        }
    }
}

pub fn init(cfg: &Telemetry, env: &str, log_level: &str) -> anyhow::Result<Guard> {
    let sentry = cfg.sentry_dsn.as_ref().map(|dsn| {
        let mut opts = sentry::ClientOptions::default();
        opts.release = sentry::release_name!();
        opts.environment = Some(env.to_owned().into());
        sentry::init((dsn.expose_secret(), opts))
    });

    let otel = match &cfg.otlp_endpoint {
        Some(endpoint) => {
            let mut builder = SpanExporter::builder().with_tonic().with_endpoint(endpoint);
            if let (Some(token), Some(dataset)) = (&cfg.axiom_token, &cfg.axiom_dataset) {
                let mut meta = tonic::metadata::MetadataMap::new();
                meta.insert(
                    "authorization",
                    format!("Bearer {}", token.expose_secret()).parse()?,
                );
                meta.insert("x-axiom-dataset", dataset.parse()?);
                builder = builder.with_metadata(meta);
            }
            let provider = SdkTracerProvider::builder()
                .with_batch_exporter(builder.build()?)
                .with_resource(
                    Resource::builder()
                        .with_service_name("sparkyai")
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
    let otel_layer = otel
        .as_ref()
        .map(|p| tracing_opentelemetry::layer().with_tracer(p.tracer("sparkyai")));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt)
        .with(sentry_tracing::layer())
        .with(otel_layer)
        .init();

    Ok(Guard {
        _sentry: sentry,
        otel,
    })
}
