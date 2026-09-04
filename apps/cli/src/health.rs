//! Periodic probes of the engine, the chat model server, and Phoenix.

use std::time::Duration;

use tokio::sync::mpsc::UnboundedSender;

use crate::core::config::Config;
use crate::core::types::{Event, Health, Probe};

/// What to probe and how often.
#[derive(Debug, Clone)]
pub struct Targets {
    engine: String,
    model: String,
    phoenix: String,
    every: Duration,
}

impl Targets {
    /// Targets from settings.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            engine: format!("{}/health", cfg.engine.base_url.trim_end_matches('/')),
            model: format!("{}/models", cfg.model.base_url.trim_end_matches('/')),
            phoenix: cfg.cli.phoenix_url.clone(),
            every: Duration::from_secs(cfg.cli.health_interval_secs.max(1)),
        }
    }
}

/// Probes forever on the configured interval, sending each result to the UI.
pub async fn poll(targets: Targets, tx: UnboundedSender<Event>) {
    let Ok(http) = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(4))
        .build()
    else {
        return;
    };
    loop {
        let (engine, model, phoenix) = tokio::join!(
            probe(&http, &targets.engine),
            probe(&http, &targets.model),
            probe(&http, &targets.phoenix)
        );
        if tx
            .send(Event::Health(Health {
                engine,
                model,
                phoenix,
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
            let body = r.text().await.unwrap_or_default();
            Probe::Degraded(format!(
                "{status}: {}",
                body.chars().take(120).collect::<String>()
            ))
        }
        Err(_) => Probe::Down,
    }
}
