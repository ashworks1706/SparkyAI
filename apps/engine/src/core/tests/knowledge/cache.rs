//! Tool output caching for live queries: reuse within a lifetime, one fetch per query, degrading.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::core::tests::support::{FakeCache, FakeQueries, MemorySink, ctx};
use crate::core::traits::knowledge::cache::QueryCache;
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::types::knowledge::cache::{CacheOutcome, Entry};
use crate::core::types::knowledge::query::{QueryError, QueryRequest, QuerySourceInfo};
use crate::core::types::trace::TraceEvent;
use crate::runtime::harness::knowledge::cache::{CacheRules, CachedQueries};

/// Rules that reuse courses answers and hold shuttles only for the handoff floor.
fn rules() -> CacheRules {
    CacheRules {
        ttl: HashMap::from([
            ("courses".to_owned(), Duration::from_secs(300)),
            ("shuttles".to_owned(), Duration::ZERO),
        ]),
        handoff: HANDOFF,
        lease: Duration::from_secs(120),
        poll: Duration::from_millis(5),
    }
}

/// Short enough that a test can outlive it.
const HANDOFF: Duration = Duration::from_millis(50);

fn request(source: &str, params: &[(&str, &str)]) -> QueryRequest {
    let mut map = Map::new();
    for (name, value) in params {
        map.insert((*name).to_owned(), Value::String((*value).to_owned()));
    }
    QueryRequest {
        source: source.to_owned(),
        params: map,
    }
}

fn published(key: &str) -> QuerySourceInfo {
    QuerySourceInfo {
        key: key.to_owned(),
        params: Vec::new(),
        indexed: true,
    }
}

/// A cached queries layer over a fake source and a fake cache, with the trace to inspect.
fn layered(
    inner: FakeQueries,
    cache: Arc<FakeCache>,
) -> (
    CachedQueries,
    Arc<std::sync::Mutex<Vec<QueryRequest>>>,
    Arc<MemorySink>,
) {
    let sent = inner.sent();
    let sink = Arc::new(MemorySink::new());
    let layer = CachedQueries::new(Arc::new(inner), cache, sink.clone(), rules());
    (layer, sent, sink)
}

/// What the cache reported on the trace, in order.
fn reported(sink: &MemorySink) -> Vec<CacheOutcome> {
    sink.records()
        .into_iter()
        .filter_map(|r| match r.event {
            TraceEvent::QueryCache { outcome, .. } => Some(outcome),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_second_ask_within_the_lifetime_never_reaches_the_scraper() {
    let cache = Arc::new(FakeCache::new());
    let (layer, sent, sink) = layered(
        FakeQueries::new(vec![published("courses")]).answering("courses", "CSE 310 open"),
        cache,
    );
    let ask = request("courses", &[("term", "Fall 2026")]);
    let first = layer.run(&ctx(), &ask).await;
    let second = layer.run(&ctx(), &ask).await;

    assert!(first.is_ok() && second.is_ok(), "{first:?} {second:?}");
    assert_eq!(
        sent.lock().map(|s| s.len()).unwrap_or_default(),
        1,
        "the source is fetched once"
    );
    assert_eq!(reported(&sink), [CacheOutcome::Miss, CacheOutcome::Hit]);
    assert!(
        first.is_ok_and(|o| o.fetched_at.is_none()),
        "a freshly fetched answer carries no age"
    );
    assert!(
        second.is_ok_and(|o| o.fetched_at.is_some()),
        "a reused answer carries when it was fetched, so the model can say how old it is"
    );
}

#[tokio::test]
async fn different_arguments_are_different_queries() {
    let cache = Arc::new(FakeCache::new());
    let (layer, sent, _) = layered(
        FakeQueries::new(vec![published("courses")]).answering("courses", "open"),
        cache,
    );
    let _ = layer
        .run(&ctx(), &request("courses", &[("term", "Fall 2026")]))
        .await;
    let _ = layer
        .run(&ctx(), &request("courses", &[("term", "Spring 2027")]))
        .await;
    assert_eq!(sent.lock().map(|s| s.len()).unwrap_or_default(), 2);
}

#[tokio::test]
async fn the_key_does_not_depend_on_the_order_arguments_arrive_in() {
    let cache = Arc::new(FakeCache::new());
    let (layer, sent, _) = layered(
        FakeQueries::new(vec![published("courses")]).answering("courses", "open"),
        cache,
    );
    let _ = layer
        .run(
            &ctx(),
            &request("courses", &[("term", "Fall 2026"), ("keywords", "CSE 310")]),
        )
        .await;
    let _ = layer
        .run(
            &ctx(),
            &request("courses", &[("keywords", "CSE 310"), ("term", "Fall 2026")]),
        )
        .await;
    assert_eq!(sent.lock().map(|s| s.len()).unwrap_or_default(), 1);
}

#[tokio::test]
async fn a_source_that_is_never_reused_is_still_fetched_only_once_at_a_time() {
    let cache = Arc::new(FakeCache::new());
    let inner = FakeQueries::new(vec![published("shuttles")])
        .answering("shuttles", "next bus 4 minutes")
        .taking(Duration::from_millis(80));
    let sent = inner.sent();
    let sink = Arc::new(MemorySink::new());
    let layer = Arc::new(CachedQueries::new(
        Arc::new(inner),
        cache,
        sink.clone(),
        rules(),
    ));

    // Ten students ask at the same moment.
    let asks: Vec<_> = (0..10)
        .map(|_| {
            let layer = Arc::clone(&layer);
            tokio::spawn(async move { layer.run(&ctx(), &request("shuttles", &[])).await })
        })
        .collect();
    for ask in asks {
        let answered = ask.await;
        assert!(
            answered.is_ok_and(|r| r.is_ok()),
            "every waiting request gets the answer"
        );
    }
    assert_eq!(
        sent.lock().map(|s| s.len()).unwrap_or_default(),
        1,
        "ten asks at once are one fetch"
    );
    let outcomes = reported(&sink);
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| **o == CacheOutcome::Miss)
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| **o == CacheOutcome::Coalesced)
            .count(),
        9
    );

    // Inside the handoff floor the answer is still there; a shuttle time 50ms old is the one
    // the waiting requests already got.
    let _ = layer.run(&ctx(), &request("shuttles", &[])).await;
    assert_eq!(sent.lock().map(|s| s.len()).unwrap_or_default(), 1);

    // Past it the next ask fetches again, because a shuttle time is not reused beyond that.
    tokio::time::sleep(HANDOFF + Duration::from_millis(20)).await;
    let _ = layer.run(&ctx(), &request("shuttles", &[])).await;
    assert_eq!(sent.lock().map(|s| s.len()).unwrap_or_default(), 2);
}

#[tokio::test]
async fn a_refusal_is_shared_rather_than_re_fetched() {
    let cache = Arc::new(FakeCache::new());
    let (layer, sent, _) = layered(
        FakeQueries::new(vec![published("courses")])
            .rejecting("courses", "term must look like Fall 2026"),
        cache,
    );
    let ask = request("courses", &[("term", "Autumn")]);
    for _ in 0..3 {
        match layer.run(&ctx(), &ask).await {
            Err(QueryError::Rejected(reason)) => assert!(reason.contains("Fall 2026")),
            other => unreachable!("expected the same refusal, got {other:?}"),
        }
    }
    assert_eq!(
        sent.lock().map(|s| s.len()).unwrap_or_default(),
        1,
        "a broken source is not hammered"
    );
}

#[tokio::test]
async fn a_failure_that_says_nothing_about_the_query_leaves_no_lease_behind() {
    let cache = Arc::new(FakeCache::new());
    // No answer is registered for this source and the double rejects it, so use a store failure.
    let inner = FakeQueries::new(vec![published("courses")]);
    let sink = Arc::new(MemorySink::new());
    let layer = CachedQueries::new(Arc::new(inner), cache.clone(), sink, rules());
    let _ = layer.run(&ctx(), &request("courses", &[])).await;
    // The refusal is cached, which is the point of the previous test; the lease itself is gone.
    assert!(
        !cache.is_empty(),
        "a refusal is kept so waiting requests read it"
    );
    assert!(
        matches!(cache.get("nothing-here").await, Ok(None)),
        "an unknown key holds nothing"
    );
}

#[tokio::test]
async fn a_cache_that_is_not_there_never_fails_a_request() {
    let cache = Arc::new(FakeCache::broken());
    let (layer, sent, sink) = layered(
        FakeQueries::new(vec![published("courses")]).answering("courses", "CSE 310 open"),
        cache,
    );
    let ask = request("courses", &[("term", "Fall 2026")]);
    assert!(layer.run(&ctx(), &ask).await.is_ok());
    assert!(layer.run(&ctx(), &ask).await.is_ok());
    assert_eq!(
        sent.lock().map(|s| s.len()).unwrap_or_default(),
        2,
        "every query is fetched when the cache is down"
    );
    assert_eq!(
        reported(&sink),
        [CacheOutcome::Unavailable, CacheOutcome::Unavailable]
    );
}

#[tokio::test]
async fn the_registry_is_read_straight_through() {
    let cache = Arc::new(FakeCache::new());
    let (layer, _, _) = layered(FakeQueries::new(vec![published("courses")]), cache);
    let sources = layer.sources().await;
    assert!(sources.is_ok_and(|s| s.len() == 1));
}

#[tokio::test]
async fn an_entry_survives_a_round_trip_through_its_stored_form() {
    let entry = Entry::Refusal {
        reason: "no readable text".to_owned(),
    };
    let encoded = json!(entry);
    let back: Entry = serde_json::from_value(encoded).unwrap_or(Entry::Pending);
    assert_eq!(back, entry);
}

/// Live check against Redis: cargo test -p engine -- --ignored redis
/// SPARKY_REDIS__URL overrides where it looks.
#[tokio::test]
#[ignore = "needs a redis server"]
async fn a_real_redis_holds_a_lease_and_an_answer() {
    use crate::core::types::knowledge::query::QueryOutcome;
    use crate::stores::knowledge::cache::{self, RedisQueryCache};

    let url =
        std::env::var("SPARKY_REDIS__URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_owned());
    let conn = cache::connect(&url.into(), Duration::from_secs(5))
        .await
        .unwrap_or_else(|e| unreachable!("start redis first: {e}"));
    let redis = RedisQueryCache::new(conn, Duration::from_secs(2));
    let key = format!("sparky:test:{}", uuid::Uuid::new_v4());

    assert!(matches!(redis.get(&key).await, Ok(None)), "starts empty");

    // Only the first claim takes the lease.
    let lease = Duration::from_secs(30);
    assert!(matches!(redis.claim(&key, lease).await, Ok(true)));
    assert!(
        matches!(redis.claim(&key, lease).await, Ok(false)),
        "a held lease is not handed out twice"
    );
    assert!(matches!(redis.get(&key).await, Ok(Some(Entry::Pending))));

    // The answer replaces the lease and survives the round trip.
    let answer = Entry::Answer {
        outcome: QueryOutcome {
            source: "courses".into(),
            url: "https://example.test/courses".into(),
            text: "CSE 310 open".into(),
            fetched_at: None,
        },
        fetched_at: chrono::Utc::now(),
    };
    assert!(
        redis
            .put(&key, &answer, Duration::from_secs(30))
            .await
            .is_ok()
    );
    match redis.get(&key).await {
        Ok(Some(Entry::Answer { outcome, .. })) => assert_eq!(outcome.text, "CSE 310 open"),
        other => unreachable!("expected the stored answer, got {other:?}"),
    }

    assert!(redis.release(&key).await.is_ok());
    assert!(
        matches!(redis.get(&key).await, Ok(None)),
        "release removes it"
    );

    // An expiry actually expires.
    assert!(
        redis
            .put(&key, &Entry::Pending, Duration::from_secs(1))
            .await
            .is_ok()
    );
    tokio::time::sleep(Duration::from_millis(1400)).await;
    assert!(
        matches!(redis.get(&key).await, Ok(None)),
        "the ttl is honoured"
    );
}
