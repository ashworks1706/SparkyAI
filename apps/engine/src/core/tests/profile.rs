//! Profile extraction: what the classifier gates, what the graph agent produces, and what
//! happens when the model answers with something neither can read.

use std::sync::Arc;

use crate::agent::harness::profile::{GRAPH_INSTRUCTIONS, GraphAgent};
use crate::agent::harness::task::{Task, TaskConfig};
use crate::core::tests::support::{Scripted, ctx, text};
use crate::core::types::context::RequestContext;
use crate::core::types::model::{ModelError, ModelResponse};
use crate::core::types::profile::ProfileError;

fn graph_agent(replies: Vec<Result<ModelResponse, ModelError>>) -> GraphAgent {
    GraphAgent::new(Task::new(
        Arc::new(Scripted::new(replies)),
        "profile_graph",
        GRAPH_INSTRUCTIONS,
        TaskConfig::default(),
    ))
}

#[tokio::test]
async fn the_graph_agent_turns_a_turn_into_facts() {
    let g = graph_agent(vec![Ok(text(
        r#"{"facts":[
             {"subject":{"kind":"person","label":"the user"},
              "relation":"studies",
              "object":{"kind":"course","label":"CSE 310"},
              "confidence":0.9},
             {"subject":{"kind":"person","label":"the user"},
              "relation":"belongs_to",
              "object":{"kind":"club","label":"AI Society"}}
           ]}"#,
    ))]);
    let Ok(facts) = g
        .extract(&ctx(), "i'm taking CSE 310 and i'm in the AI Society")
        .await
    else {
        unreachable!("the answer parses")
    };
    assert_eq!(facts.len(), 2);
    assert_eq!(facts[0].subject.label, "the user");
    assert_eq!(facts[0].relation, "studies");
    assert_eq!(facts[0].object.kind, "course");
    assert_eq!(facts[0].object.label, "CSE 310");
    assert!(facts[0].confidence < 1.0);
    // A fact that says nothing about confidence is stored as certain, not as zero.
    assert!(facts[1].confidence > 0.99);
}

#[tokio::test]
async fn an_answer_around_the_json_still_parses() {
    let g = graph_agent(vec![Ok(text(
        "```json\n{\"facts\":[{\"subject\":{\"kind\":\"person\",\"label\":\"the user\"},\
         \"relation\":\"lives_in\",\"object\":{\"kind\":\"place\",\"label\":\"Tempe\"}}]}\n```",
    ))]);
    let Ok(facts) = g.extract(&ctx(), "i live in tempe").await else {
        unreachable!("the fenced answer parses")
    };
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].object.label, "Tempe");
}

#[tokio::test]
async fn an_unparseable_extraction_is_an_error_rather_than_no_facts() {
    let g = graph_agent(vec![Ok(text(
        "Sure, I found a couple of things about them.",
    ))]);
    // An empty extraction here would silently drop everything the turn stated.
    let Err(ProfileError::Malformed(_)) = g.extract(&ctx(), "i'm taking CSE 310").await else {
        unreachable!("an unreadable answer is malformed")
    };
}

#[tokio::test]
async fn json_that_is_not_the_extraction_shape_is_an_error() {
    let g = graph_agent(vec![Ok(text(r#"{"answer":"they study CSE 310"}"#))]);
    let Err(ProfileError::Malformed(_)) = g.extract(&ctx(), "i'm taking CSE 310").await else {
        unreachable!("the wrong shape is malformed")
    };
}

#[tokio::test]
async fn a_failed_model_call_is_reported_as_a_model_error() {
    let g = graph_agent(Vec::new());
    let Err(ProfileError::Model(_)) = g.extract(&ctx(), "i'm taking CSE 310").await else {
        unreachable!("the script is exhausted")
    };
}

#[test]
fn a_relation_reads_as_a_sentence_in_the_prompt() {
    use crate::core::types::memory::{Memory, MemoryKind};
    use crate::core::types::profile::{ProfileEntity, ProfileRelation};

    let relation = ProfileRelation {
        subject: ProfileEntity {
            kind: "person".into(),
            label: "the student".into(),
        },
        relation: "studies".into(),
        object: ProfileEntity {
            kind: "subject".into(),
            label: "computer science".into(),
        },
        confidence: 0.9,
    };
    assert_eq!(relation.to_string(), "the student studies computer science");

    // A node says what a user is connected to. Only the relation says how.
    let memory = Memory::from(&relation);
    assert_eq!(memory.kind, MemoryKind::Semantic);
    assert_eq!(memory.content, "the student studies computer science");
    assert!((memory.confidence - 0.9).abs() < f32::EPSILON);
}

#[test]
fn a_recalled_node_reads_as_its_kind_and_label() {
    use crate::core::types::memory::{Memory, MemoryKind};
    use crate::core::types::profile::ProfileNode;

    let node = ProfileNode {
        id: uuid::Uuid::new_v4(),
        kind: "interest".into(),
        label: "robotics".into(),
        confidence: 1.0,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    let memory = Memory::from(&node);
    assert_eq!(memory.kind, MemoryKind::Profile);
    assert_eq!(memory.content, "interest: robotics");
}

/// Records which forget the route asked for.
#[derive(Default)]
struct Forgetful {
    calls: std::sync::Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl crate::core::traits::profile::ProfileGraph for Forgetful {
    async fn upsert(
        &self,
        _ctx: &RequestContext,
        _facts: &[crate::core::types::profile::ProfileFact],
    ) -> Result<(), crate::core::types::profile::ProfileError> {
        Ok(())
    }

    async fn recall(
        &self,
        _ctx: &RequestContext,
        _limit: usize,
    ) -> Result<Vec<crate::core::types::profile::ProfileNode>, ProfileError> {
        Ok(Vec::new())
    }

    async fn relations(
        &self,
        _ctx: &RequestContext,
        _limit: usize,
    ) -> Result<Vec<crate::core::types::profile::ProfileRelation>, ProfileError> {
        Ok(Vec::new())
    }

    async fn forget(&self, ctx: &RequestContext, label: &str) -> Result<u64, ProfileError> {
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(format!("forget {} for {}", label, ctx.user_id));
        }
        Ok(1)
    }

    async fn forget_all(&self, ctx: &RequestContext) -> Result<u64, ProfileError> {
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(format!("forget_all for {}", ctx.user_id));
        }
        Ok(3)
    }
}

#[tokio::test]
async fn a_label_removes_one_thing_and_no_label_removes_everything() {
    use axum::extract::State;
    use axum::http::HeaderMap;
    use secrecy::SecretString;

    use crate::routes::profile::{ForgetRequest, ProfileState, forget};

    let graph = std::sync::Arc::new(Forgetful::default());
    let state = ProfileState {
        graph: Some(graph.clone()),
        default_tenant: "g".into(),
        service_token: SecretString::from("t"),
    };
    let mut headers = HeaderMap::new();
    let Ok(value) = "Bearer t".parse() else {
        unreachable!("a header value")
    };
    headers.insert("authorization", value);

    forget(
        State(state.clone()),
        headers.clone(),
        axum::Json(ForgetRequest {
            user_id: "u".into(),
            tenant_id: None,
            label: Some("robotics".into()),
        }),
    )
    .await;
    forget(
        State(state),
        headers,
        axum::Json(ForgetRequest {
            user_id: "u".into(),
            tenant_id: None,
            label: None,
        }),
    )
    .await;

    let calls = graph.calls.lock().map(|c| c.clone()).unwrap_or_default();
    assert_eq!(calls, vec!["forget robotics for u", "forget_all for u"]);
}

#[tokio::test]
async fn forgetting_without_the_token_is_refused() {
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode};
    use secrecy::SecretString;

    use crate::routes::profile::{ForgetRequest, ProfileState, forget};

    let graph = std::sync::Arc::new(Forgetful::default());
    let state = ProfileState {
        graph: Some(graph.clone()),
        default_tenant: "g".into(),
        service_token: SecretString::from("t"),
    };
    let response = forget(
        State(state),
        HeaderMap::new(),
        axum::Json(ForgetRequest {
            user_id: "u".into(),
            tenant_id: None,
            label: None,
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(
        graph.calls.lock().is_ok_and(|c| c.is_empty()),
        "nothing was deleted"
    );
}
