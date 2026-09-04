//! Logging and Sentry for the bot. Traces are the API's job.

use crate::core::config::Telemetry;
use secrecy::ExposeSecret;
use tracing_subscriber::{EnvFilter, Layer, layer::SubscriberExt, util::SubscriberInitExt};

/// Keeps Sentry alive; flushes on drop.
pub struct Guard {
    _sentry: Option<sentry::ClientInitGuard>,
}

/// Installs the global `tracing` subscriber with fmt and Sentry layers.
pub fn init(cfg: &Telemetry, env: &str, log_level: &str) -> Guard {
    let sentry = cfg.sentry_dsn.as_ref().map(|dsn| {
        let mut opts = sentry::ClientOptions::default();
        opts.release = sentry::release_name!();
        opts.environment = Some(env.to_owned().into());
        sentry::init((dsn.expose_secret(), opts))
    });
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    let fmt = if env == "development" {
        tracing_subscriber::fmt::layer().pretty().boxed()
    } else {
        tracing_subscriber::fmt::layer().json().boxed()
    };
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt)
        .with(sentry_tracing::layer())
        .init();
    Guard { _sentry: sentry }
}
