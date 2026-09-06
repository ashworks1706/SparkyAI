//! The `query_source` tool: what the model is shown, and how a worker's answer or refusal reads.

use std::sync::Arc;

use serde_json::json;

use crate::agent::tools::query_source::{QuerySourceTool, describe};
use crate::core::tests::support::{FakeQueries, ctx};
use crate::core::traits::tool::Tool;
use crate::core::types::query::{QueryParam, QuerySourceInfo};
use crate::core::types::tool::{RiskClass, ToolError};

fn class_search() -> QuerySourceInfo {
    QuerySourceInfo {
        key: "class_search".into(),
        description: "Search the live ASU class catalog.".into(),
        params: vec![
            QueryParam {
                name: "term".into(),
                description: "Term to search.".into(),
                required: true,
                example: Some("Fall 2026".into()),
            },
            QueryParam {
                name: "keywords".into(),
                description: "Course number or title.".into(),
                required: false,
                example: None,
            },
        ],
    }
}

fn tool(queries: FakeQueries) -> QuerySourceTool {
    QuerySourceTool::new(Arc::new(queries), &[class_search()], 90)
}

#[test]
fn every_source_is_one_schema_and_its_parameters_are_described_in_prose() {
    let text = describe(&[class_search()]);
    assert!(text.contains("`class_search`"), "{text}");
    assert!(text.contains("term (required)"), "{text}");
    assert!(text.contains("e.g. Fall 2026"), "{text}");
    assert!(text.contains("keywords"), "{text}");
    assert!(
        !text.contains("keywords (required)"),
        "optional parameters are not marked required: {text}"
    );

    // Adding sources must not add schemas: that cost is paid on every step of every request.
    let definition = tool(FakeQueries::new(vec![class_search()])).definition();
    let props = &definition.parameters["properties"];
    assert_eq!(props.as_object().map(serde_json::Map::len), Some(2));
    assert_eq!(props["source"]["enum"], json!(["class_search"]));
    assert_eq!(definition.risk, RiskClass::ReadPublic);
    assert_eq!(
        definition.timeout_secs,
        Some(90),
        "a live fetch outlives the default tool budget"
    );
}

#[tokio::test]
async fn an_answer_carries_the_url_it_came_from() {
    let queries = FakeQueries::new(vec![class_search()]).answering("class_search", "CSE 310 open");
    let out = tool(queries)
        .call(
            &ctx(),
            json!({"source": "class_search", "params": {"term": "Fall 2026"}}),
        )
        .await;
    let Ok(output) = out else {
        unreachable!("the source answered")
    };
    assert!(
        output.content.contains("CSE 310 open"),
        "{}",
        output.content
    );
    // Without the URL the model can state a fact it cannot attribute.
    assert!(
        output.content.contains("https://example.test/class_search"),
        "{}",
        output.content
    );
}

#[tokio::test]
async fn a_refusal_comes_back_as_something_the_model_can_fix() {
    let queries = FakeQueries::new(vec![class_search()])
        .rejecting("class_search", "class_search needs: term");
    let err = tool(queries)
        .call(&ctx(), json!({"source": "class_search", "params": {}}))
        .await;
    // InvalidArguments is fed back for the model to correct; Failed would end the attempt.
    match err {
        Err(ToolError::InvalidArguments(reason)) => assert!(reason.contains("needs: term")),
        other => unreachable!("expected a correctable refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn an_unknown_source_is_refused_rather_than_fetched() {
    let queries = FakeQueries::new(vec![class_search()]);
    let err = tool(queries)
        .call(&ctx(), json!({"source": "made_up", "params": {}}))
        .await;
    assert!(
        matches!(err, Err(ToolError::InvalidArguments(_))),
        "{err:?}"
    );
}
