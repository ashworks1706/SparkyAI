//! Context assembly: ordering and budget trimming.

use chrono::{Duration, Utc};
use uuid::Uuid;

use crate::core::tests::support::ctx;
use crate::core::types::agent::assemble::{Budget, Sections, Templates};
use crate::core::types::conversation::message::{Message, Role};
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::knowledge::route::{Route, Skipped};
use crate::core::types::memory::{Memory, MemoryKind};
use crate::runtime::harness::agent::prompt::assemble::{age, assemble};

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
            date: "Friday 11 September 2026",
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
            date: "Friday 11 September 2026",
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
            date: "Friday 11 September 2026",
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
            date: "Friday 11 September 2026",
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

#[test]
fn a_resumed_run_appends_no_input_of_its_own() {
    // A resumed run adds no user turn after the history.
    let history = vec![
        Message::user("ban that spammer"),
        Message::assistant("working on it"),
    ];
    let out = assemble(
        &ctx(),
        &Sections {
            system: "sys",
            memory: &[],
            evidence: &[],
            route: Route::Retrieve,
            history: &history,
            turn: &[],
            capabilities: "",
            input: "",
            date: "Friday 11 September 2026",
            now: None,
            templates: Templates::default(),
        },
        Budget::default(),
    );

    assert_eq!(
        out.messages.last().map(|m| m.role),
        Some(Role::Assistant),
        "the last message is the history's own, not an empty user turn"
    );
    assert!(
        out.messages
            .iter()
            .all(|m| !(m.role == Role::User && m.content.is_empty())),
        "no empty user turn is added"
    );
}

#[test]
fn the_wording_around_every_section_comes_from_configuration() {
    let ev = evidence(1);
    let memories = vec![Memory {
        id: Uuid::new_v4(),
        kind: MemoryKind::Profile,
        content: "prefers mornings".into(),
        confidence: 1.0,
        created_at: Utc::now(),
        expires_at: None,
    }];
    let out = assemble(
        &ctx(),
        &Sections {
            system: "s",
            memory: &memories,
            evidence: &ev,
            input: "q",
            date: "Friday 11 September 2026",
            templates: Templates {
                role_line_no_roles: "caller {user}",
                memory_header: "REMEMBERED",
                evidence_header: "SOURCES",
                ..Templates::default()
            },
            ..Sections::default()
        },
        Budget::default(),
    );
    let text: String = out.messages.iter().map(|m| m.content.clone()).collect();
    assert!(text.contains("REMEMBERED"), "{text}");
    assert!(text.contains("SOURCES"), "{text}");
    assert!(text.contains("caller u"), "{text}");
    assert!(!text.contains("Knowledge base results"), "{text}");
}

#[test]
fn a_question_retrieval_found_nothing_for_says_so_instead_of_going_quiet() {
    let empty = assemble(
        &ctx(),
        &Sections {
            system: "s",
            input: "q",
            templates: Templates {
                no_evidence_line: "NOTHING FOUND",
                ..Templates::default()
            },
            ..Sections::default()
        },
        Budget::default(),
    );
    let text: String = empty.messages.iter().map(|m| m.content.clone()).collect();
    assert!(text.contains("NOTHING FOUND"), "{text}");
    assert_eq!(empty.evidence_used, 0);

    let ev = evidence(1);
    let found = assemble(
        &ctx(),
        &Sections {
            system: "s",
            evidence: &ev,
            input: "q",
            templates: Templates {
                no_evidence_line: "NOTHING FOUND",
                ..Templates::default()
            },
            ..Sections::default()
        },
        Budget::default(),
    );
    let text: String = found.messages.iter().map(|m| m.content.clone()).collect();
    assert!(!text.contains("NOTHING FOUND"), "{text}");
}

#[test]
fn a_turn_the_router_skipped_is_told_why_rather_than_that_nothing_was_found() {
    let templates = Templates {
        no_evidence_line: "NOTHING FOUND",
        no_retrieval_line: "NOT SEARCHED",
        live_only_line: "TOO OLD TO HELP",
        ..Templates::default()
    };
    let line = |route| {
        let out = assemble(
            &ctx(),
            &Sections {
                system: "s",
                input: "q",
                route,
                templates,
                ..Sections::default()
            },
            Budget::default(),
        );
        out.messages
            .iter()
            .map(|m| m.content.clone())
            .collect::<String>()
    };
    assert!(line(Route::Retrieve).contains("NOTHING FOUND"));
    let chitchat = line(Route::Skip(Skipped::Chitchat));
    assert!(chitchat.contains("NOT SEARCHED"), "{chitchat}");
    assert!(
        !chitchat.contains("NOTHING FOUND"),
        "a skipped turn is never told retrieval came back empty"
    );
    let current = line(Route::Skip(Skipped::Live));
    assert!(current.contains("TOO OLD TO HELP"), "{current}");
    assert!(!current.contains("NOTHING FOUND"), "{current}");
}

#[test]
fn an_evidence_entry_carries_its_number_its_page_and_its_date() {
    let mut ev = evidence(1);
    ev[0].url = Some("https://lib.asu.edu/hours".into());
    let out = assemble(
        &ctx(),
        &Sections {
            system: "s",
            evidence: &ev,
            input: "q",
            ..Sections::default()
        },
        Budget::default(),
    );
    let text: String = out.messages.iter().map(|m| m.content.clone()).collect();
    assert!(
        text.contains("[1] Doc 0 - https://lib.asu.edu/hours"),
        "{text}"
    );
    assert!(text.contains("(stored copy, fetched "), "{text}");
}

#[test]
fn the_capabilities_heading_is_written_around_the_rendered_list() {
    let out = assemble(
        &ctx(),
        &Sections {
            system: "s",
            capabilities: "- search_library_hours (tool): finds ASU pages",
            input: "q",
            templates: Templates {
                capabilities_header: "WHAT YOU CAN DO",
                ..Templates::default()
            },
            ..Sections::default()
        },
        Budget::default(),
    );
    let text: String = out.messages.iter().map(|m| m.content.clone()).collect();
    assert!(
        text.contains("WHAT YOU CAN DO\n- search_library_hours (tool)"),
        "{text}"
    );
}

#[test]
fn a_finer_tokenizer_estimate_fits_less_into_the_same_budget() {
    // Two characters per token fits less evidence than eight in the same budget.
    let ev = evidence(20);
    let count = |chars_per_token: usize| {
        assemble(
            &ctx(),
            &Sections {
                system: "s",
                evidence: &ev,
                input: "q",
                date: "Friday 11 September 2026",
                ..Sections::default()
            },
            Budget {
                evidence: 800,
                chars_per_token,
                ..Budget::default()
            },
        )
        .evidence_used
    };
    assert!(count(2) < count(8), "{} < {}", count(2), count(8));
}

#[test]
fn the_date_reaches_the_prompt_so_today_is_answerable() {
    // The date appears in the assembled prompt.
    let a = assemble(
        &ctx(),
        &Sections {
            system: "You are Sparky.",
            input: "hours today?",
            date: "Friday 11 September 2026",
            ..Sections::default()
        },
        Budget::default(),
    );
    let system = &a.messages[0].content;
    assert!(system.contains("Friday 11 September 2026"), "{system}");
}

#[test]
fn no_date_writes_no_date_line() {
    let a = assemble(
        &ctx(),
        &Sections {
            system: "You are Sparky.",
            input: "hi",
            date: "",
            ..Sections::default()
        },
        Budget::default(),
    );
    assert!(
        !a.messages[0].content.contains("Today is"),
        "{:?}",
        a.messages[0]
    );
}

#[test]
fn thinking_the_model_wrote_inline_is_lifted_out_of_the_answer() {
    use crate::runtime::harness::agent::call::thought::split;

    let (thought, visible) = split(
        "",
        "<think>The hours are not in what I was given.</think>Hayden closes at 2am.",
    );
    assert_eq!(
        thought.as_deref(),
        Some("The hours are not in what I was given.")
    );
    assert_eq!(
        visible, "Hayden closes at 2am.",
        "the tags never reach a user"
    );

    // A step that ran out of room mid-thought still reported a thought.
    let (cut, visible) = split("", "<think>I should search for");
    assert_eq!(cut.as_deref(), Some("I should search for"));
    assert!(visible.is_empty());

    // Reasoning the provider returned in a field of its own wins over the text.
    let (given, visible) = split("provider reasoning", "<think>inline</think>answer");
    assert_eq!(given.as_deref(), Some("provider reasoning"));
    assert_eq!(visible, "answer");

    let (none, visible) = split("", "Hayden closes at 2am.");
    assert!(none.is_none());
    assert_eq!(visible, "Hayden closes at 2am.");
}

#[test]
fn the_tool_exchange_of_this_request_follows_the_question_and_is_never_trimmed() {
    use crate::core::types::conversation::message::ToolCall;

    let call = Message::assistant_tool_calls(
        "",
        vec![ToolCall {
            id: "c1".into(),
            name: "search_news".into(),
            arguments: serde_json::json!({"keywords": "robotics"}),
        }],
    );
    let result = Message::tool_result("c1", "search_news", "x".repeat(4_000));
    let turn = [call, result];
    let ev = evidence(20);
    let out = assemble(
        &ctx(),
        &Sections {
            system: "s",
            evidence: &ev,
            input: "latest news",
            turn: &turn,
            date: "Friday 11 September 2026",
            ..Sections::default()
        },
        Budget {
            total: 1_200,
            ..Budget::default()
        },
    );
    let roles: Vec<Role> = out.messages.iter().map(|m| m.role).collect();
    let asked = out.messages.iter().position(|m| m.content == "latest news");
    let answered = out.messages.iter().position(|m| m.role == Role::Tool);
    assert!(
        asked.is_some() && answered > asked,
        "the result comes after the question: {roles:?}"
    );
    assert_eq!(
        out.messages.last().map(|m| m.role),
        Some(Role::Tool),
        "the tool result is the last thing the model reads"
    );
    assert_eq!(
        out.evidence_used, 0,
        "evidence gives way to the result of this turn when the budget is tight"
    );
}

#[test]
fn every_evidence_entry_says_it_is_a_stored_copy_and_how_old_it_is() {
    let now = Utc::now();
    let mut ev = evidence(1);
    if let Some(first) = ev.first_mut() {
        first.fetched_at = now - Duration::days(3);
    }
    let out = assemble(
        &ctx(),
        &Sections {
            system: "s",
            evidence: &ev,
            input: "q",
            now: Some(now),
            ..Sections::default()
        },
        Budget::default(),
    );
    let text: String = out.messages.iter().map(|m| m.content.clone()).collect();
    assert!(text.contains("(stored copy, fetched "), "{text}");
    assert!(text.contains(", 3 days ago)"), "{text}");
    assert!(text.contains("call the matching search_ tool"), "{text}");
}

#[test]
fn a_page_age_reads_in_hours_under_two_days_and_in_days_after() {
    let now = Utc::now();
    assert_eq!(age(now - Duration::minutes(10), now), "under an hour ago");
    assert_eq!(age(now - Duration::minutes(90), now), "1 hour ago");
    assert_eq!(age(now - Duration::hours(30), now), "30 hours ago");
    assert_eq!(age(now - Duration::hours(50), now), "2 days ago");
}
