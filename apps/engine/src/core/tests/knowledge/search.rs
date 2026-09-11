//! The category filter of search_knowledge_base: what it offers and what it refuses.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use crate::agent::tools::knowledge::search::KnowledgeSearch;
use crate::core::traits::knowledge::retrieval::Retriever;
use crate::core::traits::tools::Tool;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::knowledge::retrieval::{RetrievalError, RetrievalQuery};
use crate::core::types::tools::ToolError;

fn ctx() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5))
}

/// Answers every query with one chunk, whatever was asked for.
struct Always;

#[async_trait]
impl Retriever for Always {
    async fn retrieve(
        &self,
        _ctx: &RequestContext,
        _query: &RetrievalQuery,
    ) -> Result<Vec<Evidence>, RetrievalError> {
        Ok(vec![Evidence {
            source_id: Uuid::new_v4(),
            chunk_id: Uuid::new_v4(),
            title: "ASU Library hours".into(),
            content: "Sep 11 Friday 7am - 10pm".into(),
            url: None,
            fetched_at: Utc::now(),
            score: 1.0,
        }])
    }
}

fn tool() -> KnowledgeSearch {
    KnowledgeSearch::new(Arc::new(Always), 3, vec!["events".into(), "library".into()])
}

#[test]
fn the_indexed_categories_are_the_only_values_the_filter_offers() {
    let d = tool().definition();
    let values = &d.parameters["properties"]["categories"]["items"]["enum"];
    assert_eq!(values, &serde_json::json!(["events", "library"]));
    // A model that cannot see the categories guesses one, and a guess matches nothing.
    assert!(
        d.description.contains("events, library"),
        "{}",
        d.description
    );
}

#[tokio::test]
async fn a_category_the_index_does_not_hold_is_refused_with_the_ones_it_does() {
    // Returning no results for a category that does not exist reads as an absent fact.
    let out = tool()
        .call(
            &ctx(),
            serde_json::json!({"query": "hours", "categories": ["library_hours"]}),
        )
        .await;
    let Err(ToolError::InvalidArguments(message)) = out else {
        unreachable!("a category that does not exist is a correctable mistake, got {out:?}")
    };
    assert!(message.contains("library_hours"), "{message}");
    assert!(message.contains("events, library"), "{message}");
}

#[tokio::test]
async fn an_indexed_category_searches_and_case_does_not_matter() {
    for asked in ["library", "Library"] {
        let out = tool()
            .call(
                &ctx(),
                serde_json::json!({"query": "hours", "categories": [asked]}),
            )
            .await;
        let Ok(out) = out else {
            unreachable!("{asked} is indexed, got {out:?}")
        };
        assert!(out.content.contains("Sep 11 Friday"), "{}", out.content);
    }
}

#[tokio::test]
async fn omitting_the_filter_searches_every_category() {
    let out = tool()
        .call(&ctx(), serde_json::json!({"query": "hours"}))
        .await;
    assert!(out.is_ok(), "{out:?}");
}
