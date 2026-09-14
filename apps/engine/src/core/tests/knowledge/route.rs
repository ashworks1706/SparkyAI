//! The gate in front of retrieval: which turns skip it, and what the loop does with a skip.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use crate::core::tests::support::{agent_routing, ctx, text};
use crate::core::traits::knowledge::retrieval::Retriever;
use crate::core::traits::knowledge::route::Router;
use crate::core::types::agent::AgentConfig;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::agent::thinking::{ThinkingMode, ThinkingReason};
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::knowledge::retrieval::{RetrievalError, RetrievalQuery};
use crate::core::types::knowledge::route::{Route, Skipped};
use crate::core::types::trace::TraceEvent;
use crate::runtime::harness::agent::call::thinking::{StepSignals, decide};
use crate::runtime::harness::knowledge::route::{RuleRouter, Rules};

fn routed(input: &str) -> Route {
    RuleRouter::default().route(input)
}

/// Counts the retrieval calls it is asked for and answers with one chunk.
#[derive(Default)]
struct Counting(AtomicUsize);

#[async_trait]
impl Retriever for Counting {
    async fn retrieve(
        &self,
        _ctx: &RequestContext,
        _query: &RetrievalQuery,
    ) -> Result<Vec<Evidence>, RetrievalError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(vec![Evidence {
            source_id: Uuid::new_v4(),
            chunk_id: Uuid::new_v4(),
            title: "hayden hours".into(),
            content: "Hayden closes at midnight.".into(),
            url: None,
            fetched_at: Utc::now(),
            score: 1.0,
        }])
    }
}

#[test]
fn a_question_about_an_asu_fact_is_retrieved_for() {
    for asked in [
        "any AI clubs",
        "what scholarships can a junior in CS apply for",
        "how do I appeal a parking ticket",
        "who runs the AI Society",
    ] {
        assert_eq!(routed(asked), Route::Retrieve, "{asked}");
    }
}

#[test]
fn a_cue_that_the_answer_must_be_current_skips_retrieval() {
    for asked in [
        "when is the next shuttle to Poly",
        "are there study rooms free at hayden",
        "what is the score right now",
        "is hayden open now",
        "search the web for the ASU game",
    ] {
        assert_eq!(routed(asked), Route::Skip(Skipped::Live), "{asked}");
    }
}

#[test]
fn small_talk_skips_retrieval_but_a_question_carrying_a_marker_does_not() {
    for asked in ["hey", "thanks!", "what can you do", "ok cool", "  "] {
        assert_eq!(routed(asked), Route::Skip(Skipped::Chitchat), "{asked}");
    }
    // Past max_chitchat_words the marker no longer decides.
    let long = "hey do you know which clubs run machine learning workshops";
    assert_eq!(routed(long), Route::Retrieve, "{long}");
    // A word that also reads as part of an ASU question is not a marker.
    for asked in ["is it ok to eat in hayden", "no parking on campus"] {
        assert_eq!(routed(asked), Route::Retrieve, "{asked}");
    }
}

#[test]
fn a_live_cue_beats_a_chitchat_marker() {
    assert_eq!(routed("hey whats the score"), Route::Skip(Skipped::Live));
}

#[test]
fn a_marker_matches_whole_words_only() {
    // "hi" inside "history", "no" inside "nobel".
    assert_eq!(routed("history of the AI Society"), Route::Retrieve);
    assert_eq!(routed("where is nobel library"), Route::Retrieve);
}

#[test]
fn configured_lists_replace_the_built_in_ones() {
    let rules = Rules::from(&crate::core::config::Router {
        enabled: true,
        max_chitchat_words: 4,
        chitchat: vec!["salve".into()],
        live: vec!["A la minute".into()],
    });
    let router = RuleRouter::new(rules);
    assert_eq!(router.route("salve"), Route::Skip(Skipped::Chitchat));
    assert_eq!(
        router.route("hours a la minute"),
        Route::Skip(Skipped::Live)
    );
    assert_eq!(
        router.route("thanks"),
        Route::Retrieve,
        "built-ins are gone"
    );
}

/// A router that always decides the same way.
struct Fixed(Route);

impl Router for Fixed {
    fn route(&self, _input: &str) -> Route {
        self.0
    }
}

#[tokio::test]
async fn a_skipped_turn_never_reaches_the_retriever_and_says_why_in_the_prompt() {
    use crate::core::tests::support::Scripted;

    let counter = Arc::new(Counting::default());
    let (agent, sink) = agent_routing(
        Scripted::new(vec![Ok(text("hello"))]),
        AgentConfig::default(),
        counter.clone(),
        Arc::new(Fixed(Route::Skip(Skipped::Chitchat))),
    );
    let answer = agent.run(&ctx(), "hey").await;
    assert!(answer.is_ok(), "{answer:?}");
    assert_eq!(
        counter.0.load(Ordering::SeqCst),
        0,
        "the retriever is not called"
    );
    let events: Vec<TraceEvent> = sink.records().into_iter().map(|r| r.event).collect();
    assert!(
        events.iter().any(|e| matches!(
            e,
            TraceEvent::RetrievalSkipped {
                reason: Skipped::Chitchat
            }
        )),
        "the skip is recorded"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, TraceEvent::Retrieval { .. })),
        "no retrieval event is recorded"
    );
}

#[tokio::test]
async fn a_routed_turn_still_retrieves() {
    use crate::core::tests::support::Scripted;

    let counter = Arc::new(Counting::default());
    let (agent, _) = agent_routing(
        Scripted::new(vec![Ok(text("answered"))]),
        AgentConfig::default(),
        counter.clone(),
        Arc::new(Fixed(Route::Retrieve)),
    );
    let answer = agent.run(&ctx(), "any AI clubs").await;
    assert!(answer.is_ok(), "{answer:?}");
    assert_eq!(counter.0.load(Ordering::SeqCst), 1);
}

#[test]
fn a_skipped_turn_does_not_spend_the_thinking_budget_on_missing_evidence() {
    let mut rules = AgentConfig::default().thinking;
    rules.mode = ThinkingMode::Auto;
    let signals = |retrieved| StepSignals {
        answer_only: false,
        tool_results: false,
        input: "hey",
        retrieved,
        evidence: 0,
    };
    assert_eq!(
        decide(&rules, &signals(true)).reason,
        ThinkingReason::NoEvidence
    );
    let skipped = decide(&rules, &signals(false));
    assert!(!skipped.on, "a skipped turn answers without reasoning");
    assert_eq!(skipped.reason, ThinkingReason::Quick);
}
