//! Health probes for the engine, chat model, and PostHog, plus the port-in-use check.

use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use tokio::sync::mpsc::UnboundedSender;

use crate::core::config::Config;
use crate::core::types::{Event, Health, Probe};

/// What to probe and how often.
#[derive(Debug, Clone)]
pub struct Targets {
    engine: String,
    model: String,
    posthog: String,
    every: Duration,
}

impl Targets {
    /// Targets from settings.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            engine: format!("{}/health/ready", cfg.engine.base_url.trim_end_matches('/')),
            model: format!("{}/models", cfg.model.base_url.trim_end_matches('/')),
            posthog: format!("{}/_health", cfg.cli.posthog_url.trim_end_matches('/')),
            every: Duration::from_secs(cfg.cli.health_interval_secs.max(1)),
        }
    }
}

/// Probes forever on the configured interval, sending each result to the UI.
pub async fn poll(targets: Targets, tx: UnboundedSender<Event>) {
    let http = match reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(4))
        .build()
    {
        Ok(http) => http,
        Err(e) => {
            let failed = Probe::Degraded(format!("health client: {e}"));
            // A send error means the UI has exited.
            let _ = tx.send(Event::Health(Health {
                engine: failed.clone(),
                model: failed.clone(),
                posthog: failed,
            }));
            return;
        }
    };
    loop {
        let (engine, model, posthog) = tokio::join!(
            probe(&http, &targets.engine),
            probe(&http, &targets.model),
            probe(&http, &targets.posthog)
        );
        if tx
            .send(Event::Health(Health {
                engine,
                model,
                posthog,
            }))
            .is_err()
        {
            return;
        }
        tokio::time::sleep(targets.every).await;
    }
}

async fn probe(http: &reqwest::Client, url: &str) -> Probe {
    match http.get(url).send().await {
        Ok(r) if r.status().is_success() => Probe::Up,
        Ok(r) => {
            let status = r.status();
            let body = r
                .text()
                .await
                .unwrap_or_else(|e| format!("unreadable body: {e}"));
            Probe::Degraded(format!(
                "{status}: {}",
                body.chars().take(120).collect::<String>()
            ))
        }
        Err(_) => Probe::Down,
    }
}

/// The host and port a unit's url names, if reachable. Wildcard host maps to loopback.
pub fn address_of(url: &str) -> Option<String> {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let rest = rest.split(['/', '?', '#']).next()?;
    let (host, port) = rest.rsplit_once(':')?;
    if port.is_empty() || port.parse::<u16>().is_err() {
        return None;
    }
    let host = match host.trim() {
        "" | "0.0.0.0" | "[::]" => "127.0.0.1",
        other => other,
    };
    Some(format!("{host}:{port}"))
}

/// Whether something already accepts connections at addr.
pub fn served(addr: &str, timeout: Duration) -> bool {
    let Ok(resolved) = addr.to_socket_addrs() else {
        return false;
    };
    resolved
        .into_iter()
        .any(|target| TcpStream::connect_timeout(&target, timeout).is_ok())
}
