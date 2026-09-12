//! Thinking per model call: the rules that decide it and the retry without it.

use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::agent::harness::agent::call::thinking::{StepSignals, decide};
use crate::agent::harness::tools::ToolSet;
use crate::core::tests::support::{Echo, Scripted, agent, calls, ctx, only_thought, text};
use crate::core::types::agent::AgentConfig;
use crate::core::types::agent::thinking::{ThinkingMode, ThinkingReason, ThinkingRules};
use crate::core::types::model::ModelRequest;
use crate::core::types::tools::RiskClass;
use crate::core::types::trace::{RunStatus, TraceEvent};

fn rules() -> ThinkingRules {
    AgentConfig::default().thinking
}

/// A first step on input with evidence found.
fn first(input: &str, evidence: usize) -> StepSignals<'_> {
    StepSignals {
        answer_only: false,
        tool_results: false,
        input,
        evidence,
    }
}

/// Whether each request the model was sent asked it to think.
fn thinking_sent(sent: &Arc<Mutex<Vec<ModelRequest>>>) -> Vec<bool> {
    sent.lock()
        .map(|requests| requests.iter().map(|r| r.thinking).collect())
        .unwrap_or_default()
}

#[test]
fn a_short_question_with_evidence_answers_without_thinking() {
    let choice = decide(&rules(), &first("when does hayden close tonight", 3));
    assert!(!choice.on);
    assert_eq!(choice.reason, ThinkingReason::Quick);
}

#[test]
fn a_question_retrieval_found_nothing_for_thinks() {
    let choice = decide(&rules(), &first("when does hayden close tonight", 0));
    assert!(choice.on);
    assert_eq!(choice.reason, ThinkingReason::NoEvidence);
}

#[test]
fn a_cue_matches_whole_words_in_any_case() {
    let asked = decide(&rules(), &first("Why is the shuttle late", 3));
    assert_eq!((asked.on, asked.reason), (true, ThinkingReason::Cue));
    let inside = decide(&rules(), &first("showhow tickets", 3));
    assert_eq!(
        inside.reason,
        ThinkingReason::Quick,
        "a cue inside a word is not a cue"
    );
    let phrase = ThinkingRules {
        cues: vec!["Pros and cons".into()],
        ..rules()
    };
    let matched = decide(&phrase, &first("pros and cons of the meal plan", 3));
    assert_eq!(matched.reason, ThinkingReason::Cue);
}

#[test]
fn a_long_question_thinks() {
    let long = "tell me about ".repeat(10);
    let choice = decide(&rules(), &first(&long, 3));
    assert_eq!((choice.on, choice.reason), (true, ThinkingReason::Long));
}

#[test]
fn tool_results_and_answer_only_steps_come_before_the_question_rules() {
    let after = decide(
        &rules(),
        &StepSignals {
            tool_results: true,
            ..first("when", 3)
        },
    );
    assert_eq!((after.on, after.reason), (true, ThinkingReason::AfterTools));
    let quiet = ThinkingRules {
        after_tools: false,
        ..rules()
    };
    let after = decide(
        &quiet,
        &StepSignals {
            tool_results: true,
            ..first("why", 0)
        },
    );
    assert!(!after.on);
    let forced = decide(
        &rules(),
        &StepSignals {
            answer_only: true,
            tool_results: true,
            ..first("why", 0)
        },
    );
    assert_eq!(
        (forced.on, forced.reason),
        (false, ThinkingReason::AnswerOnly)
    );
}

#[test]
fn a_fixed_mode_ignores_every_rule() {
    for (mode, on) in [(ThinkingMode::On, true), (ThinkingMode::Off, false)] {
        let fixed = ThinkingRules { mode, ..rules() };
        for step in [first("hi", 3), first("why", 0)] {
            let choice = decide(&fixed, &step);
            assert_eq!((choice.on, choice.reason), (on, ThinkingReason::Mode));
        }
    }
}

#[tokio::test]
async fn the_loop_sends_the_decision_with_each_call() {
    let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
    let model = Scripted::new(vec![
        Ok(calls(vec![("c1", "echo", json!({}))])),
        Ok(text("done")),
    ]);
    let sent = model.sent();
    let cfg = AgentConfig {
        thinking: ThinkingRules {
            after_tools: false,
            ..rules()
        },
        ..AgentConfig::default()
    };
    let (agent, _) = agent(model, tools, cfg);
    assert!(agent.run(&ctx(), "when").await.is_ok());
    assert_eq!(
        thinking_sent(&sent),
        vec![true, false],
        "no evidence thinks, then tool results with after_tools off do not"
    );
}

#[tokio::test]
async fn thinking_that_leaves_no_answer_is_asked_again_without_it() {
    let model = Scripted::new(vec![
        Ok(only_thought("the hours are in row two, and")),
        Ok(text("Hayden closes at 2am.")),
    ]);
    let sent = model.sent();
    let (agent, sink) = agent(model, ToolSet::new(), AgentConfig::default());
    let out = agent.run(&ctx(), "when does hayden close").await.ok();
    assert_eq!(
        out.as_ref().map(|answer| answer.text.as_str()),
        Some("Hayden closes at 2am.")
    );
    assert_eq!(out.as_ref().map(|answer| answer.steps), Some(1), "one step");
    assert_eq!(
        out.map(|answer| answer.usage.completion_tokens),
        Some(10),
        "both calls are counted"
    );
    assert_eq!(thinking_sent(&sent), vec![true, false]);
    let started = sink
        .records()
        .iter()
        .filter(|record| matches!(record.event, TraceEvent::ModelStarted { .. }))
        .count();
    assert_eq!(started, 1, "the retry does not open a second thinking line");
}

#[tokio::test]
async fn with_the_retry_off_a_spent_step_says_it_ran_out_of_room() {
    let model = Scripted::new(vec![Ok(only_thought("still reasoning"))]);
    let sent = model.sent();
    let cfg = AgentConfig {
        thinking: ThinkingRules {
            retry_without: false,
            ..rules()
        },
        ..AgentConfig::default()
    };
    let (agent, _) = agent(model, ToolSet::new(), cfg);
    let out = agent.run(&ctx(), "when does hayden close").await.ok();
    assert_eq!(thinking_sent(&sent), vec![true]);
    assert!(
        out.as_ref()
            .is_some_and(|answer| answer.text.contains("ran out of room"))
    );
    assert_eq!(out.map(|answer| answer.status), Some(RunStatus::Answered));
}
