//! Live queries on the platform: the registry and a synchronous fetch.

use std::time::Duration;

use serde_json::{Map, json};

use super::{Fake, caller, client};
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::query::{QueryError, QueryRequest};
use crate::stores::platform::PlatformQueries;

fn request(source: &str) -> QueryRequest {
    let mut params = Map::new();
    params.insert("campus".into(), json!("tempe"));
    QueryRequest {
        source: source.into(),
        params,
    }
}

#[tokio::test]
async fn the_registry_reads_keys_and_params_and_counts_every_source_indexed() {
    let fake = Fake::default();
    fake.on(
        "GET",
        "/api/asu/queries",
        200,
        json!({"queries": [{
            "key": "library_hours",
            "description": "Hours",
            "params": [{"name": "campus", "description": "d", "required": true,
                        "example": "tempe", "choices": ["tempe", "west"], "many": false}]
        }]}),
    );
    let Ok(sources) = PlatformQueries::new(client(&fake).await).sources().await else {
        unreachable!("the registry loads")
    };
    assert_eq!(sources[0].key, "library_hours");
    assert!(sources[0].params[0].required);
    assert_eq!(sources[0].params[0].choices, vec!["tempe", "west"]);
    assert!(sources[0].indexed);
}

#[tokio::test]
async fn a_query_returns_the_cited_page() {
    let fake = Fake::default();
    fake.on(
        "POST",
        "/api/asu/query",
        200,
        json!({"source": "library_hours", "url": "https://lib.asu.edu/hours", "text": "Open 7 to 2"}),
    );
    let Ok(outcome) = PlatformQueries::new(client(&fake).await)
        .run(&caller(), &request("library_hours"))
        .await
    else {
        unreachable!("the query answers")
    };
    assert_eq!(
        fake.last().body,
        json!({"source": "library_hours", "params": {"campus": "tempe"}})
    );
    assert_eq!(outcome.url, "https://lib.asu.edu/hours");
    assert_eq!(outcome.text, "Open 7 to 2");
    assert!(outcome.fetched_at.is_none());
}

#[tokio::test]
async fn a_refused_query_goes_back_to_the_model() {
    let fake = Fake::default();
    fake.on(
        "POST",
        "/api/asu/query",
        422,
        json!({"error": "campus must be one of tempe, west"}),
    );
    let result = PlatformQueries::new(client(&fake).await)
        .run(&caller(), &request("library_hours"))
        .await;
    assert!(matches!(result, Err(QueryError::Rejected(m)) if m.contains("campus")));
}

#[tokio::test]
async fn a_bad_token_is_a_store_failure_not_a_rejection() {
    let fake = Fake::default();
    fake.on(
        "POST",
        "/api/asu/query",
        403,
        json!({"error": "missing scope"}),
    );
    let result = PlatformQueries::new(client(&fake).await)
        .run(&caller(), &request("library_hours"))
        .await;
    assert!(matches!(result, Err(QueryError::Store(_))));
}

#[tokio::test]
async fn a_slow_query_stops_at_the_request_deadline() {
    let fake = Fake::default();
    fake.after(
        "POST",
        "/api/asu/query",
        200,
        json!({"source": "s", "url": "u", "text": "t"}),
        Duration::from_secs(5),
    );
    let ctx = RequestContext::new("guild", "111", Duration::from_millis(200));
    let result = PlatformQueries::new(client(&fake).await)
        .run(&ctx, &request("s"))
        .await;
    assert!(matches!(result, Err(QueryError::Timeout(_))));
}

#[tokio::test]
async fn a_cancelled_request_cancels_the_query() {
    let fake = Fake::default();
    fake.after(
        "POST",
        "/api/asu/query",
        200,
        json!({"source": "s", "url": "u", "text": "t"}),
        Duration::from_secs(5),
    );
    let ctx = caller();
    let cancel = ctx.cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        cancel.cancel();
    });
    let result = PlatformQueries::new(client(&fake).await)
        .run(&ctx, &request("s"))
        .await;
    assert!(matches!(result, Err(QueryError::Cancelled)));
}
