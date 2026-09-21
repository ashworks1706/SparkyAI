//! The engine sandbox seen from the console: what it is running, and the two ways to stop it.

use std::collections::HashMap;
use std::time::Duration;

use secrecy::ExposeSecret as _;
use secrecy::SecretString;
use tokio::sync::mpsc::UnboundedSender;

use crate::core::config::Config;
use crate::core::types::{Event, SandboxCommand, SandboxReport};

/// Where the sandbox routes are and what they need.
#[derive(Debug, Clone)]
pub struct Endpoint {
    base: String,
    token: SecretString,
    every: Duration,
}

impl Endpoint {
    /// The endpoint from settings.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            base: cfg.engine.base_url.trim_end_matches('/').to_owned(),
            token: cfg.engine.service_token.clone(),
            every: Duration::from_secs(cfg.cli.health_interval_secs.max(1)),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    /// The path of one container, with the name escaped so it can only name a container.
    fn container_url(&self, container: &str) -> String {
        let escaped: String = container
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                    c.to_string()
                } else {
                    format!("%{:02X}", c as u32 & 0xff)
                }
            })
            .collect();
        self.url(&format!("/sandbox/{escaped}"))
    }
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(4))
        .build()
        .map_err(|e| format!("sandbox client: {e}"))
}

/// Reads the sandbox forever on the configured interval, sending each report to the UI.
pub async fn poll(endpoint: Endpoint, tx: UnboundedSender<Event>) {
    let http = match client() {
        Ok(http) => http,
        Err(e) => {
            let _ = tx.send(Event::Sandbox(Err(e)));
            return;
        }
    };
    loop {
        if tx
            .send(Event::Sandbox(report(&http, &endpoint).await))
            .is_err()
        {
            return;
        }
        tokio::time::sleep(endpoint.every).await;
    }
}

async fn report(http: &reqwest::Client, endpoint: &Endpoint) -> Result<SandboxReport, String> {
    let sent = http
        .get(endpoint.url("/sandbox"))
        .bearer_auth(endpoint.token.expose_secret())
        .send()
        .await
        .map_err(|e| format!("engine unreachable: {e}"))?;
    if !sent.status().is_success() {
        return Err(format!("engine answered {}", sent.status()));
    }
    sent.json::<SandboxReport>()
        .await
        .map_err(|e| format!("unreadable sandbox report: {e}"))
}

/// The commands still to write, oldest first, given the end state already written for each.
///
/// A command is written once as it starts and once as it ends; a repeated state writes nothing.
pub fn unwritten(commands: &[SandboxCommand], shown: &HashMap<u64, bool>) -> Vec<SandboxCommand> {
    commands
        .iter()
        .rev()
        .filter(|c| shown.get(&c.id) != Some(&c.ended()))
        .cloned()
        .collect()
}

/// Removes one session container. Its workspace goes with it.
pub async fn kill(endpoint: Endpoint, container: String) -> Result<String, String> {
    let http = client()?;
    let sent = http
        .delete(endpoint.container_url(&container))
        .bearer_auth(endpoint.token.expose_secret())
        .send()
        .await
        .map_err(|e| format!("engine unreachable: {e}"))?;
    if sent.status().is_success() {
        return Ok(format!("killed {container}"));
    }
    Err(format!("{container} was not killed: {}", sent.status()))
}

/// Offers the sandbox to the agent, or stops offering it.
pub async fn switch(endpoint: Endpoint, on: bool) -> Result<String, String> {
    let http = client()?;
    let sent = http
        .post(endpoint.url("/sandbox/enabled"))
        .bearer_auth(endpoint.token.expose_secret())
        .json(&serde_json::json!({ "enabled": on }))
        .send()
        .await
        .map_err(|e| format!("engine unreachable: {e}"))?;
    if sent.status().is_success() {
        return Ok(if on {
            "the agent may use the sandbox".to_owned()
        } else {
            "the agent is no longer offered the sandbox".to_owned()
        });
    }
    Err(format!("the switch was refused: {}", sent.status()))
}
