use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;

use crate::agent::model::limit::Limited;
use crate::core::tests::support::{ctx, text};
use crate::core::traits::model::ModelProvider;
use crate::core::types::context::RequestContext;
use crate::core::types::model::{ModelError, ModelRequest, ModelResponse};

/// Records how many calls were ever in flight at once.
struct Counting {
    inflight: AtomicU32,
    peak: AtomicU32,
    delay: Duration,
}

impl Counting {
    fn new(delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            inflight: AtomicU32::new(0),
            peak: AtomicU32::new(0),
            delay,
        })
    }
}

#[async_trait]
impl ModelProvider for Counting {
    async fn generate(
        &self,
        _ctx: &RequestContext,
        _req: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        let now = self.inflight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.inflight.fetch_sub(1, Ordering::SeqCst);
        Ok(text("ok"))
    }
}

fn request() -> ModelRequest {
    ModelRequest {
        messages: Vec::new(),
        tools: Vec::new(),
        max_tokens: 16,
        temperature: 0.0,
    }
}

async fn call_all(limited: Arc<Limited>, n: usize) -> Vec<Result<ModelResponse, ModelError>> {
    let mut handles = Vec::new();
    for _ in 0..n {
        let limited = limited.clone();
        handles.push(tokio::spawn(async move {
            limited.generate(&ctx(), request()).await
        }));
    }
    let mut out = Vec::new();
    for h in handles {
        out.push(h.await.unwrap_or_else(|e| Err(ModelError::Transport(e.to_string()))));
    }
    out
}

#[tokio::test]
async fn calls_never_exceed_the_slot_count() {
    let inner = Counting::new(Duration::from_millis(60));
    let limited = Arc::new(Limited::new(
        inner.clone(),
        2,
        Duration::from_secs(5),
    ));

    let results = call_all(limited, 5).await;

    assert!(results.iter().all(Result::is_ok), "every call should be served");
    assert_eq!(inner.peak.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_call_that_waits_past_its_budget_reports_busy() {
    let inner = Counting::new(Duration::from_millis(300));
    let limited = Arc::new(Limited::new(
        inner,
        1,
        Duration::from_millis(30),
    ));

    let results = call_all(limited, 2).await;

    let busy = results
        .iter()
        .filter(|r| matches!(r, Err(ModelError::Busy)))
        .count();
    assert_eq!(busy, 1, "one call is served, the other gives up waiting");
}

#[tokio::test]
async fn cancelling_while_queued_ends_the_call() {
    let inner = Counting::new(Duration::from_millis(300));
    let limited = Arc::new(Limited::new(inner, 1, Duration::from_secs(5)));

    let first = {
        let limited = limited.clone();
        tokio::spawn(async move { limited.generate(&ctx(), request()).await })
    };
    tokio::time::sleep(Duration::from_millis(30)).await;

    let queued = ctx();
    let cancel = queued.cancel.clone();
    let second = tokio::spawn(async move { limited.generate(&queued, request()).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    cancel.cancel();

    assert!(matches!(
        second.await.unwrap_or(Ok(text("unreached"))),
        Err(ModelError::Cancelled)
    ));
    assert!(first.await.is_ok());
}

#[test]
fn busy_is_not_retried() {
    assert!(!ModelError::Busy.is_retryable());
}
