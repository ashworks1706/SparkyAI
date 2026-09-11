//! Product events for PostHog: a bounded queue fed without waiting and a background task
//! that posts batches to host plus /batch/.

use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::core::config::{Analytics as Settings, Telemetry};
use crate::core::telemetry::export_target;
use crate::core::types::{AnalyticsBatch, AnalyticsEvent};

/// Path of the PostHog batch capture endpoint under the host.
pub const BATCH_PATH: &str = "/batch/";

/// Handle that queues product events. Cheap to clone; a disabled handle drops everything.
#[derive(Debug, Clone)]
pub struct Analytics {
    tx: Option<mpsc::Sender<AnalyticsEvent>>,
}

/// Stops the background sender and flushes what it holds.
#[derive(Debug)]
pub struct Flusher {
    stop: oneshot::Sender<()>,
    task: JoinHandle<()>,
}

impl Analytics {
    /// A handle that records nothing.
    pub fn disabled() -> Self {
        Self { tx: None }
    }

    /// A handle and the receiving end of its queue, with nothing draining it.
    pub fn channel(capacity: usize) -> (Self, mpsc::Receiver<AnalyticsEvent>) {
        let (tx, rx) = mpsc::channel(capacity);
        (Self { tx: Some(tx) }, rx)
    }

    /// Starts the background sender. Disabled when settings are off or telemetry has no
    /// host or project token.
    pub fn start(settings: &Settings, telemetry: &Telemetry) -> (Self, Option<Flusher>) {
        if !settings.enabled {
            tracing::info!("analytics disabled by analytics.enabled");
            return (Self::disabled(), None);
        }
        let Some((host, token)) = export_target(telemetry) else {
            tracing::info!("analytics disabled; telemetry host or project token is empty");
            return (Self::disabled(), None);
        };
        let http = match reqwest::Client::builder()
            .timeout(Duration::from_secs(telemetry.export_timeout_secs))
            .build()
        {
            Ok(http) => http,
            Err(e) => {
                tracing::warn!(error = %e, "analytics client failed to build; analytics off");
                return (Self::disabled(), None);
            }
        };
        let sink = Sink {
            http,
            url: format!("{host}{BATCH_PATH}"),
            token: token.clone(),
        };
        let (handle, rx) = Self::channel(settings.queue_capacity);
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(drain(
            rx,
            stopped,
            sink,
            settings.max_batch,
            Duration::from_millis(settings.flush_ms),
        ));
        (handle, Some(Flusher { stop, task }))
    }

    /// Queues event without waiting. Returns false when it was dropped.
    pub fn record(&self, event: AnalyticsEvent) -> bool {
        let Some(tx) = &self.tx else {
            return false;
        };
        match tx.try_send(event) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(event)) => {
                tracing::warn!(event = event.event, "analytics queue full; event dropped");
                false
            }
            Err(mpsc::error::TrySendError::Closed(event)) => {
                tracing::debug!(event = event.event, "analytics stopped; event dropped");
                false
            }
        }
    }
}

impl Flusher {
    /// Sends what is queued and stops, waiting at most within.
    pub async fn finish(self, within: Duration) {
        let _ = self.stop.send(());
        if tokio::time::timeout(within, self.task).await.is_err() {
            tracing::warn!("analytics flush timed out; events dropped");
        }
    }
}

/// Where batches go.
struct Sink {
    http: reqwest::Client,
    url: String,
    token: SecretString,
}

impl Sink {
    /// Posts one batch. A failure drops it and logs.
    async fn send(&self, batch: Vec<AnalyticsEvent>) {
        if batch.is_empty() {
            return;
        }
        let body = AnalyticsBatch {
            api_key: self.token.expose_secret(),
            batch: &batch,
        };
        match self.http.post(&self.url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!(events = batch.len(), "analytics batch sent");
            }
            Ok(resp) => {
                tracing::warn!(
                    events = batch.len(),
                    status = resp.status().as_u16(),
                    "analytics batch refused; dropped"
                );
            }
            Err(e) => {
                tracing::warn!(events = batch.len(), error = %e, "analytics batch failed; dropped");
            }
        }
    }
}

/// Batches events until max_batch or every flush_every, then flushes the rest on stop.
async fn drain(
    mut rx: mpsc::Receiver<AnalyticsEvent>,
    mut stopped: oneshot::Receiver<()>,
    sink: Sink,
    max_batch: usize,
    flush_every: Duration,
) {
    let mut batch = Vec::with_capacity(max_batch);
    let mut tick = tokio::time::interval(flush_every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            got = rx.recv() => match got {
                Some(event) => {
                    batch.push(event);
                    if batch.len() >= max_batch {
                        sink.send(std::mem::take(&mut batch)).await;
                    }
                }
                None => break,
            },
            _ = tick.tick() => sink.send(std::mem::take(&mut batch)).await,
            _ = &mut stopped => break,
        }
    }
    rx.close();
    while let Ok(event) = rx.try_recv() {
        batch.push(event);
        if batch.len() >= max_batch {
            sink.send(std::mem::take(&mut batch)).await;
        }
    }
    sink.send(batch).await;
}
