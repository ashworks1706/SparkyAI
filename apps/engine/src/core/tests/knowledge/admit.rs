//! The cap on live queries: what it refuses, what it never counts, and what it does when unread.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;

use crate::core::tests::support::{FakeCache, FakeQueries, MemorySink, ctx};
use crate::core::traits::knowledge::admission::Admission;
use crate::core::traits::knowledge::cache::QueryCache;
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::types::knowledge::cache::CacheError;
use crate::core::types::knowledge::query::{QueryError, QueryRequest, QuerySourceInfo};
use crate::core::types::trace::TraceEvent;
use crate::runtime::harness::knowledge::admit::AdmittedQueries;
use crate::runtime::harness::knowledge::cache::{CacheRules, CachedQueries};

/// A cap that lets in limit holders at a time and counts what it was asked.
struct Counted {
    limit: usize,
    held: std::sync::Mutex<Vec<String>>,
    asked: AtomicUsize,
    broken: bool,
}

impl Counted {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            held: std::sync::Mutex::new(Vec::new()),
            asked: AtomicUsize::new(0),
            broken: false,
        }
    }

    fn broken() -> Self {
        Self {
            broken: true,
            ..Self::new(0)
        }
    }

    fn asked(&self) -> usize {
        self.asked.load(Ordering::SeqCst)
    }

    fn held(&self) -> usize {
        self.held.lock().map(|h| h.len()).unwrap_or_default()
    }
}

#[async_trait]
impl Admission for Counted {
    async fn enter(&self, holder: &str) -> Result<bool, CacheError> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        if self.broken {
            return Err(CacheError::Backend("no cap here".into()));
        }
        let Ok(mut held) = self.held.lock() else {
            return Ok(false);
        };
        if held.len() >= self.limit {
            return Ok(false);
        }
        held.push(holder.to_owned());
        Ok(true)
    }

    async fn leave(&self, holder: &str) -> Result<(), CacheError> {
        if let Ok(mut held) = self.held.lock() {
            held.retain(|h| h != holder);
        }
        Ok(())
    }
}

fn published(key: &str) -> QuerySourceInfo {
    QuerySourceInfo {
        key: key.to_owned(),
        params: Vec::new(),
        indexed: true,
    }
}

fn request(source: &str) -> QueryRequest {
    QueryRequest {
        source: source.to_owned(),
        params: serde_json::Map::new(),
    }
}

#[tokio::test]
async fn a_query_over_the_cap_is_refused_rather_than_queued() {
    let cap = Arc::new(Counted::new(1));
    let inner = FakeQueries::new(vec![published("courses")])
        .answering("courses", "open")
        .taking(Duration::from_millis(60));
    let sent = inner.sent();
    let sink = Arc::new(MemorySink::new());
    let layer = Arc::new(AdmittedQueries::new(
        Arc::new(inner),
        cap.clone(),
        sink.clone(),
    ));

    // Two at once, one slot.
    let a = {
        let layer = Arc::clone(&layer);
        tokio::spawn(async move { layer.run(&ctx(), &request("courses")).await })
    };
    tokio::time::sleep(Duration::from_millis(10)).await;
    let b = layer.run(&ctx(), &request("courses")).await;

    assert!(matches!(b, Err(QueryError::Busy(source)) if source == "courses"));
    assert!(
        a.await.is_ok_and(|r| r.is_ok()),
        "the one that got in finishes"
    );
    assert_eq!(
        sent.lock().map(|s| s.len()).unwrap_or_default(),
        1,
        "the refused query never reaches the database"
    );
    assert!(
        sink.records()
            .iter()
            .any(|r| matches!(r.event, TraceEvent::QueryRefused { .. })),
        "the refusal is recorded"
    );
}

#[tokio::test]
async fn the_refusal_tells_the_model_what_to_do_instead() {
    let cap = Arc::new(Counted::new(0));
    let sink = Arc::new(MemorySink::new());
    let layer = AdmittedQueries::new(
        Arc::new(FakeQueries::new(vec![published("courses")]).answering("courses", "open")),
        cap,
        sink,
    );
    match layer.run(&ctx(), &request("courses")).await {
        Err(error) => {
            let said = error.to_string();
            assert!(
                said.contains("courses") && said.contains("ask again"),
                "{said}"
            );
        }
        other => unreachable!("expected a refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn a_slot_is_given_back_whether_the_query_worked_or_not() {
    let cap = Arc::new(Counted::new(1));
    let sink = Arc::new(MemorySink::new());
    let layer = AdmittedQueries::new(
        Arc::new(FakeQueries::new(vec![published("courses")]).rejecting("courses", "bad term")),
        cap.clone(),
        sink,
    );
    for _ in 0..3 {
        assert!(layer.run(&ctx(), &request("courses")).await.is_err());
    }
    assert_eq!(cap.held(), 0, "a failed query does not hold its slot");
    assert_eq!(cap.asked(), 3, "each attempt asked for a slot and got one");
}

#[tokio::test]
async fn a_cap_that_cannot_be_read_caps_nothing_rather_than_refusing_everything() {
    let cap = Arc::new(Counted::broken());
    let inner = FakeQueries::new(vec![published("courses")]).answering("courses", "open");
    let sent = inner.sent();
    let sink = Arc::new(MemorySink::new());
    let layer = AdmittedQueries::new(Arc::new(inner), cap, sink);
    assert!(layer.run(&ctx(), &request("courses")).await.is_ok());
    assert_eq!(sent.lock().map(|s| s.len()).unwrap_or_default(), 1);
}

#[tokio::test]
async fn a_reused_answer_takes_no_slot_and_touches_no_database() {
    // The cap sits under the cache, which is what makes a hit free.
    let cap = Arc::new(Counted::new(8));
    let inner = FakeQueries::new(vec![published("courses")]).answering("courses", "open");
    let sent = inner.sent();
    let sink = Arc::new(MemorySink::new());
    let admitted = Arc::new(AdmittedQueries::new(
        Arc::new(inner),
        cap.clone(),
        sink.clone(),
    ));
    let cache: Arc<dyn QueryCache> = Arc::new(FakeCache::new());
    let layer = CachedQueries::new(
        admitted,
        cache,
        sink,
        CacheRules {
            ttl: std::collections::HashMap::from([(
                "courses".to_owned(),
                Duration::from_secs(300),
            )]),
            handoff: Duration::from_secs(5),
            lease: Duration::from_secs(120),
            poll: Duration::from_millis(5),
        },
    );

    for _ in 0..5 {
        assert!(layer.run(&ctx(), &request("courses")).await.is_ok());
    }
    assert_eq!(
        sent.lock().map(|s| s.len()).unwrap_or_default(),
        1,
        "one fetch for five asks"
    );
    assert_eq!(
        cap.asked(),
        1,
        "only the fetch asked for a slot; the four reused answers did not"
    );
}

/// Live check against Redis: cargo test -p engine -- --ignored redis
/// SPARKY_REDIS__URL overrides where it looks.
#[tokio::test]
#[ignore = "needs a redis server"]
async fn a_real_redis_caps_slots_and_drops_holders_that_never_left() {
    use crate::stores::knowledge::cache::{self as redis_cache, RedisAdmission};

    let url =
        std::env::var("SPARKY_REDIS__URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned());
    let conn = redis_cache::connect(&url.clone().into(), Duration::from_secs(5))
        .await
        .unwrap_or_else(|e| unreachable!("start redis first: {e}"));
    let key = format!("sparky:test:cap:{}", uuid::Uuid::new_v4());
    let cap = RedisAdmission::new(
        conn,
        Duration::from_secs(2),
        key.clone(),
        2,
        Duration::from_secs(60),
    );

    // Two slots, three askers.
    assert!(matches!(cap.enter("a").await, Ok(true)));
    assert!(matches!(cap.enter("b").await, Ok(true)));
    assert!(
        matches!(cap.enter("c").await, Ok(false)),
        "the third is refused"
    );

    // Giving one back frees exactly one.
    assert!(cap.leave("a").await.is_ok());
    assert!(matches!(cap.enter("c").await, Ok(true)));
    assert!(matches!(cap.enter("d").await, Ok(false)));

    // Leaving twice is harmless, and leaving a holder that never entered frees nothing.
    assert!(cap.leave("a").await.is_ok());
    assert!(matches!(cap.enter("d").await, Ok(false)));

    // A holder older than the lease is dropped rather than holding its slot forever.
    let short = RedisAdmission::new(
        redis_cache::connect(&url.into(), Duration::from_secs(5))
            .await
            .unwrap_or_else(|e| unreachable!("{e}")),
        Duration::from_secs(2),
        format!("{key}:short"),
        1,
        Duration::from_secs(1),
    );
    assert!(matches!(short.enter("stuck").await, Ok(true)));
    assert!(matches!(short.enter("next").await, Ok(false)));
    tokio::time::sleep(Duration::from_millis(2_100)).await;
    assert!(
        matches!(short.enter("next").await, Ok(true)),
        "a slot whose holder never left is reclaimed once it is older than the lease"
    );
}
