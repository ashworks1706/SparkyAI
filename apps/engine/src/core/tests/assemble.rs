//! Context assembly: ordering and budget trimming.

use chrono::Utc;
use uuid::Uuid;

use crate::agent::harness::assemble::assemble;
use crate::core::tests::support::ctx;
use crate::core::types::assemble::{Budget, Sections};
use crate::core::types::evidence::Evidence;
use crate::core::types::message::{Message, Role};

fn evidence(n: usize) -> Vec<Evidence> {
    (0..n)
        .map(|i| Evidence {
            source_id: Uuid::new_v4(),
            chunk_id: Uuid::new_v4(),
            title: format!("Doc {i}"),
            content: "x".repeat(400),
            url: None,
            fetched_at: Utc::now(),
            score: 1.0,
        })
        .collect()
}

#[test]
fn system_comes_first_and_input_last() {
    let out = assemble(
        &ctx(),
        &Sections {
            system: "You are Sparky.",
            input: "hi",
            ..Sections::default()
        },
        Budget::default(),
    );
    assert_eq!(out.messages.first().map(|m| m.role), Some(Role::System));
    assert_eq!(out.messages.last().map(|m| m.content.as_str()), Some("hi"));
}

#[test]
fn evidence_is_trimmed_to_its_budget() {
    let ev = evidence(20);
    let out = assemble(
        &ctx(),
        &Sections {
            system: "s",
            evidence: &ev,
            input: "q",
            ..Sections::default()
        },
        Budget {
            evidence: 500,
            ..Budget::default()
        },
    );
    assert!(out.evidence_used < 20);
    assert!(out.evidence_used >= 1);
}

#[test]
fn history_keeps_the_newest_turns() {
    let history: Vec<Message> = (0..50)
        .map(|i| Message::user(format!("turn {i} {}", "y".repeat(100))))
        .collect();
    let out = assemble(
        &ctx(),
        &Sections {
            system: "s",
            history: &history,
            input: "q",
            ..Sections::default()
        },
        Budget {
            history: 300,
            ..Budget::default()
        },
    );
    assert!(out.messages.len() < 52);
    let last_history = &out.messages[out.messages.len() - 2];
    assert!(last_history.content.starts_with("turn 49"));
}

#[test]
fn history_never_starts_with_a_tool_result() {
    let history = vec![
        Message::assistant_tool_calls("", vec![]),
        Message::tool_result("c1", "echo", "big result ".repeat(50)),
        Message::user("follow-up"),
    ];
    let out = assemble(
        &ctx(),
        &Sections {
            system: "s",
            history: &history,
            input: "q",
            ..Sections::default()
        },
        Budget {
            history: 160,
            ..Budget::default()
        },
    );
    let first_history = out.messages.iter().skip(1).find(|m| m.role != Role::System);
    assert!(first_history.is_none_or(|m| m.role != Role::Tool));
}
