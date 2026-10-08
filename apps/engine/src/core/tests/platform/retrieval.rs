//! Knowledge search on the platform: the query vector comes from the engine embedder.

use std::sync::Arc;

use serde_json::json;

use super::{Fake, caller, client};
use crate::core::tests::support::FixedEmbedder;
use crate::core::traits::knowledge::retrieval::{Embedder, Retriever};
use crate::core::types::knowledge::retrieval::RetrievalQuery;
use crate::stores::platform::{PlatformRetriever, SearchTuning};

const SEARCH: &str = "/api/knowledge/search";

fn tuning() -> SearchTuning {
    SearchTuning {
        embedding_model: "embed-model".into(),
        window: 1,
        max_query_chars: 12,
    }
}

fn results() -> serde_json::Value {
    json!({"dense": true, "results": [
        {
            "chunk_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
            "source_key": "asu:library-hours",
            "title": "Library hours",
            "url": "https://lib.asu.edu/hours",
            "category": "library",
            "public": true,
            "content": "Hayden opens at 7.",
            "score": 0.032_787,
            "fetched_at": "2026-10-07T12:00:00"
        },
        {
            "chunk_id": "client-written-1",
            "source_key": "notes",
            "title": null,
            "url": null,
            "category": "club",
            "public": false,
            "content": "Meetings on Fridays.",
            "score": 0.016_129,
            "fetched_at": "2026-10-06T12:00:00"
        },
        {
            "chunk_id": "8c9e6679-7425-40de-944b-e07fc1f90ae7",
            "source_key": "undated",
            "title": "Undated",
            "url": null,
            "category": "club",
            "public": false,
            "content": "No date.",
            "score": 0.01,
            "fetched_at": null
        }
    ]})
}

#[tokio::test]
async fn the_query_vector_and_model_are_sent_and_rows_become_evidence() {
    let fake = Fake::default();
    fake.on("POST", SEARCH, 200, results());
    let embedder = FixedEmbedder::returning(vec![0.5, 0.25]);
    let asked = embedder.asked();
    let retriever = PlatformRetriever::new(
        client(&fake).await,
        Some(Arc::new(embedder) as Arc<dyn Embedder>),
        tuning(),
    );
    let query = RetrievalQuery::new("library hours today please", 4).in_category("library");
    let Ok(evidence) = retriever.retrieve(&caller(), &query).await else {
        unreachable!("search answers")
    };

    let sent = fake.last().body;
    assert_eq!(sent["query"], "library hour");
    assert_eq!(sent["top_k"], 4);
    assert_eq!(sent["window"], 1);
    assert_eq!(sent["category"], "library");
    assert_eq!(sent["embedding"], json!([0.5, 0.25]));
    assert_eq!(sent["embedding_model"], "embed-model");
    assert_eq!(
        asked.lock().map(|a| a.clone()).unwrap_or_default(),
        vec!["library hour".to_owned()]
    );

    assert_eq!(evidence.len(), 2, "a row without a fetch date is dropped");
    assert_eq!(evidence[0].key, "asu:library-hours");
    assert_eq!(evidence[0].title, "Library hours");
    assert_eq!(
        evidence[0].url.as_deref(),
        Some("https://lib.asu.edu/hours")
    );
    assert_eq!(
        evidence[0].chunk_id.to_string(),
        "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    );
    assert_eq!(
        evidence[1].title, "notes",
        "an untitled source shows its key"
    );
    assert_ne!(evidence[0].source_id, evidence[1].source_id);
    assert!(evidence[0].score > evidence[1].score);
}

#[tokio::test]
async fn one_source_key_always_names_the_same_source() {
    let fake = Fake::default();
    fake.on("POST", SEARCH, 200, results());
    let retriever = PlatformRetriever::new(client(&fake).await, None, tuning());
    let query = RetrievalQuery::new("q", 4);
    let (Ok(first), Ok(second)) = (
        retriever.retrieve(&caller(), &query).await,
        retriever.retrieve(&caller(), &query).await,
    ) else {
        unreachable!("search answers")
    };
    assert_eq!(first[1].source_id, second[1].source_id);
    assert_eq!(first[1].chunk_id, second[1].chunk_id);
}

#[tokio::test]
async fn a_failed_embedding_falls_back_to_a_text_search() {
    let fake = Fake::default();
    fake.on("POST", SEARCH, 200, json!({"dense": false, "results": []}));
    let retriever = PlatformRetriever::new(
        client(&fake).await,
        Some(Arc::new(FixedEmbedder::failing()) as Arc<dyn Embedder>),
        tuning(),
    );
    let result = retriever
        .retrieve(&caller(), &RetrievalQuery::new("q", 3))
        .await;
    assert!(matches!(result, Ok(e) if e.is_empty()));
    let sent = fake.last().body;
    assert!(sent.get("embedding").is_none());
    assert!(sent.get("embedding_model").is_none());
    assert!(sent.get("category").is_none());
}

#[tokio::test]
async fn a_platform_refusal_is_a_store_error() {
    let fake = Fake::default();
    fake.on(
        "POST",
        SEARCH,
        400,
        json!({"error": "embedding must be a list of 1024 numbers"}),
    );
    let retriever = PlatformRetriever::new(client(&fake).await, None, tuning());
    let result = retriever
        .retrieve(&caller(), &RetrievalQuery::new("q", 3))
        .await;
    assert!(matches!(result, Err(e) if e.to_string().contains("1024")));
}
