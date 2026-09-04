//! The loop: answers, tool feedback, parallelism, policy, limits, retries, cost, redaction.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use crate::agent::harness::agent::redact;
use crate::agent::harness::tool::ToolSet;
use crate::core::tests::support::{Echo, Ordered, Scripted, Slow, agent, calls, ctx, text};
use crate::core::types::agent::AgentConfig;
use crate::core::types::model::ModelError;
use crate::core::types::policy::Decision;
use crate::core::types::tool::RiskClass;
use crate::core::types::trace::{RunStatus, TraceEvent};

#[tokio::test]
async fn text_reply_is_the_answer() {
    let (agent, sink) = agent(
        Scripted::new(vec![Ok(text("2am"))]),
        ToolSet::new(),
        AgentConfig::default(),
    );
    let out = agent.run(&ctx(), "when?").await.ok();
    let out = out.as_ref();
    assert_eq!(out.map(|answer| answer.text.as_str()), Some("2am"));
    assert_eq!(out.map(|answer| answer.steps), Some(1));
    assert!(
        sink.records()
            .iter()
            .any(|record| matches!(record.event, TraceEvent::Completed { .. }))
    );
}

#[tokio::test]
async fn tool_result_is_fed_back_and_loop_continues() {
    let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
    let (agent, sink) = agent(
        Scripted::new(vec![
            Ok(calls(vec![("c1", "echo", json!({"q": 1}))])),
            Ok(text("done")),
        ]),
        tools,
        AgentConfig::default(),
    );
    let out = agent.run(&ctx(), "go").await.ok();
    assert_eq!(
        out.as_ref().map(|answer| answer.text.as_str()),
        Some("done")
    );
    assert_eq!(out.as_ref().map(|answer| answer.steps), Some(2));
    assert!(sink.records().iter().any(
        |record| matches!(&record.event, TraceEvent::ToolCall { tool, .. } if tool == "echo")
    ));
}

#[tokio::test]
async fn parallel_calls_all_run() {
    let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
    let (agent, sink) = agent(
        Scripted::new(vec![
            Ok(calls(vec![
                ("c1", "echo", json!(1)),
                ("c2", "echo", json!(2)),
                ("c3", "echo", json!(3)),
            ])),
            Ok(text("ok")),
        ]),
        tools,
        AgentConfig::default(),
    );
    let _ = agent.run(&ctx(), "go").await;
    let count = sink
        .records()
        .iter()
        .filter(|record| matches!(record.event, TraceEvent::ToolCall { .. }))
        .count();
    assert_eq!(count, 3);
}

#[tokio::test]
async fn write_without_role_is_denied_not_run() {
    let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ExternalWrite)));
    let (agent, sink) = agent(
        Scripted::new(vec![
            Ok(calls(vec![("c1", "echo", json!({}))])),
            Ok(text("ok")),
        ]),
        tools,
        AgentConfig::default(),
    );
    let _ = agent.run(&ctx(), "post it").await;
    let records = sink.records();
    assert!(records.iter().any(|record| matches!(
        &record.event,
        TraceEvent::PolicyDecision {
            decision: Decision::Deny { .. },
            ..
        }
    )));
    assert!(
        !records
            .iter()
            .any(|record| matches!(record.event, TraceEvent::ToolCall { .. }))
    );
}

#[tokio::test]
async fn write_with_role_stops_for_confirmation() {
    let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ExternalWrite)));
    let (agent, _) = agent(
        Scripted::new(vec![
            Ok(calls(vec![("c1", "echo", json!({}))])),
            Ok(text("never")),
        ]),
        tools,
        AgentConfig::default(),
    );
    let context = ctx().with_roles(vec!["MANAGE_GUILD".into()]);
    let out = agent.run(&context, "post it").await.ok();
    assert_eq!(
        out.as_ref().map(|answer| answer.status.clone()),
        Some(RunStatus::AwaitingConfirmation)
    );
    assert!(
        out.as_ref()
            .is_some_and(|answer| answer.confirmation.is_some())
    );
}

#[tokio::test]
async fn step_limit_stops_the_loop() {
    let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
    let script: Vec<_> = (0..10)
        .map(|i| Ok(calls(vec![(&format!("c{i}"), "echo", json!(i))])))
        .collect();
    let (agent, _) = agent(
        Scripted::new(script),
        tools,
        AgentConfig {
            max_steps: 3,
            ..AgentConfig::default()
        },
    );
    let out = agent.run(&ctx(), "loop").await.ok();
    assert_eq!(
        out.as_ref().map(|answer| answer.status.clone()),
        Some(RunStatus::StepLimit)
    );
    assert_eq!(out.as_ref().map(|answer| answer.steps), Some(3));
}

#[tokio::test]
async fn tool_timeout_becomes_an_error_result() {
    let tools = ToolSet::new().with(Arc::new(Slow));
    let (agent, sink) = agent(
        Scripted::new(vec![
            Ok(calls(vec![("c1", "slow", json!({}))])),
            Ok(text("ok")),
        ]),
        tools,
        AgentConfig {
            tool_timeout: Duration::from_millis(50),
            ..AgentConfig::default()
        },
    );
    let _ = agent.run(&ctx(), "go").await;
    assert!(sink.records().iter().any(|record| matches!(
        &record.event,
        TraceEvent::ToolCall { result: Err(message), .. } if message.contains("timed out")
    )));
}

#[tokio::test]
async fn cancellation_ends_the_run() {
    let (agent, _) = agent(
        Scripted::new(vec![Ok(text("x"))]),
        ToolSet::new(),
        AgentConfig::default(),
    );
    let context = ctx();
    context.cancel.cancel();
    let out = agent.run(&context, "go").await.ok();
    assert_eq!(out.map(|answer| answer.status), Some(RunStatus::Cancelled));
}

#[tokio::test]
async fn retryable_model_error_is_retried() {
    let (agent, sink) = agent(
        Scripted::new(vec![
            Err(ModelError::Transport("boom".into())),
            Ok(text("recovered")),
        ]),
        ToolSet::new(),
        AgentConfig::default(),
    );
    let out = agent.run(&ctx(), "go").await.ok();
    assert_eq!(out.map(|answer| answer.text), Some("recovered".into()));
    assert!(
        sink.records()
            .iter()
            .any(|record| matches!(record.event, TraceEvent::ModelError { retried: true, .. }))
    );
}

#[tokio::test]
async fn usage_and_cost_accumulate() {
    let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
    let (agent, _) = agent(
        Scripted::new(vec![
            Ok(calls(vec![("c1", "echo", json!({}))])),
            Ok(text("ok")),
        ]),
        tools,
        AgentConfig {
            usd_per_m_prompt: 1.0,
            usd_per_m_completion: 2.0,
            ..AgentConfig::default()
        },
    );
    let out = agent.run(&ctx(), "go").await.ok();
    let out = out.as_ref();
    assert_eq!(out.map(|answer| answer.usage.prompt_tokens), Some(20));
    assert_eq!(out.map(|answer| answer.usage.completion_tokens), Some(10));
    assert!(out.is_some_and(|answer| (answer.cost_usd - 0.000_04).abs() < 1e-12));
}

#[test]
fn secrets_are_redacted_from_traces() {
    let redacted = redact(&json!({"user": "a", "password": "b", "nested": {"api_key": "c"}}));
    assert_eq!(redacted["user"], "a");
    assert_eq!(redacted["password"], "[redacted]");
    assert_eq!(redacted["nested"]["api_key"], "[redacted]");
}

#[tokio::test]
async fn a_repeated_call_forces_a_tool_free_answer() {
    let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
    let same = || Ok(calls(vec![("c", "echo", json!({"q": 1}))]));
    // Step 1 calls, step 2 repeats, step 3 (no tools offered) answers.
    let (agent, sink) = agent(
        Scripted::new(vec![same(), same(), Ok(text("from the result"))]),
        tools,
        AgentConfig::default(),
    );
    let out = agent.run(&ctx(), "loop").await.ok();
    assert_eq!(
        out.as_ref().map(|answer| answer.text.as_str()),
        Some("from the result")
    );
    assert_eq!(out.map(|answer| answer.status), Some(RunStatus::Answered));
    let executed = sink
        .records()
        .iter()
        .filter(|record| matches!(record.event, TraceEvent::ToolCall { .. }))
        .count();
    assert_eq!(executed, 1, "the repeat must not run again");
}

#[tokio::test]
async fn repeating_even_without_tools_stalls() {
    let tools = ToolSet::new().with(Arc::new(Echo(RiskClass::ReadPublic)));
    let same = || Ok(calls(vec![("c", "echo", json!({"q": 1}))]));
    // Step 3 gets no tools; a scripted model that still returns an empty answer stalls.
    let mut empty = text("");
    empty.finish_reason = crate::core::types::model::FinishReason::Stop;
    let (agent, _) = agent(
        Scripted::new(vec![same(), same(), Ok(empty)]),
        tools,
        AgentConfig::default(),
    );
    let out = agent.run(&ctx(), "loop").await.ok();
    assert_eq!(out.map(|answer| answer.status), Some(RunStatus::Stalled));
}

#[tokio::test]
async fn stateful_tools_run_in_order() {
    let tools = ToolSet::new().with(Arc::new(Ordered(RiskClass::ReadPublic)));
    let (agent, sink) = agent(
        Scripted::new(vec![
            Ok(calls(vec![
                ("c1", "ordered", json!({"n": 1})),
                ("c2", "ordered", json!({"n": 2})),
                ("c3", "ordered", json!({"n": 3})),
            ])),
            Ok(text("ok")),
        ]),
        tools,
        AgentConfig::default(),
    );
    let _ = agent.run(&ctx(), "go").await;
    let order: Vec<String> = sink
        .records()
        .iter()
        .filter_map(|record| match &record.event {
            TraceEvent::ToolCall {
                result: Ok(text), ..
            } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(order, vec!["1", "2", "3"]);
}
