//! Profile extraction: what the classifier gates, what the graph agent produces, and what
//! happens when the model answers with something neither can read.

use std::sync::Arc;

use crate::agent::harness::profile::{
    CLASSIFIER_INSTRUCTIONS, Classifier, GRAPH_INSTRUCTIONS, GraphAgent,
};
use crate::agent::harness::task::{Task, TaskConfig};
use crate::core::tests::support::{Scripted, ctx, text};
use crate::core::types::model::{ModelError, ModelResponse};
use crate::core::types::profile::ProfileError;

fn classifier(replies: Vec<Result<ModelResponse, ModelError>>) -> Classifier {
    Classifier::new(Task::new(
        Arc::new(Scripted::new(replies)),
        "profile_classifier",
        CLASSIFIER_INSTRUCTIONS,
        TaskConfig::default(),
    ))
}

fn graph_agent(replies: Vec<Result<ModelResponse, ModelError>>) -> GraphAgent {
    GraphAgent::new(Task::new(
        Arc::new(Scripted::new(replies)),
        "profile_graph",
        GRAPH_INSTRUCTIONS,
        TaskConfig::default(),
    ))
}

#[tokio::test]
async fn small_talk_carries_nothing_to_extract() {
    let c = classifier(vec![Ok(text("no"))]);
    let Ok(carries) = c.carries_fact(&ctx(), "hey sparky, how's it going").await else {
        unreachable!("the model answered")
    };
    assert!(!carries);
}

#[tokio::test]
async fn a_stated_preference_is_worth_extracting() {
    let c = classifier(vec![Ok(text("Yes."))]);
    let Ok(carries) = c
        .carries_fact(&ctx(), "i'd rather study at hayden than noble")
        .await
    else {
        unreachable!("the model answered")
    };
    // The one word is the whole contract; casing and punctuation around it are not.
    assert!(carries);
}

#[tokio::test]
async fn an_answer_that_is_not_one_word_is_read_as_no() {
    let c = classifier(vec![Ok(text("Well, it might be a preference, I think"))]);
    let Ok(carries) = c.carries_fact(&ctx(), "something ambiguous").await else {
        unreachable!("the model answered")
    };
    // Queueing extraction on a hedge would run the graph agent on every turn.
    assert!(!carries);
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
