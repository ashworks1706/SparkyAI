//! The OpenAI-compatible surface: what an off-the-shelf chat client sends and expects back.

use crate::core::types::http::openai::ChatMessage;
use crate::routes::openai::{conversation_for, last_user_message, transcript};

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
    }
}

#[test]
fn the_newest_user_turn_is_the_input() {
    let history = vec![
        msg("system", "be helpful"),
        msg("user", "hours?"),
        msg("assistant", "7am to 2am"),
        msg("user", "and on friday?"),
    ];
    assert_eq!(last_user_message(&history), Some("and on friday?"));
    assert_eq!(last_user_message(&[msg("system", "hi")]), None);
    assert_eq!(last_user_message(&[]), None);
}

#[test]
fn one_chat_keeps_one_conversation_and_two_chats_do_not_share() {
    let first = "what are hayden hours";
    // The client resends the whole history each turn; the id must not move with it.
    assert_eq!(
        conversation_for("ash", first),
        conversation_for("ash", first)
    );
    assert_ne!(
        conversation_for("ash", first),
        conversation_for("ash", "when is the career fair")
    );
    assert_ne!(
        conversation_for("ash", first),
        conversation_for("sam", first),
        "two callers opening with the same question stay apart"
    );
}

#[test]
fn tools_and_citations_ride_along_in_the_content() {
    use crate::core::types::agent::Answer;
    use crate::core::types::knowledge::evidence::{Citation, Evidence};
    use crate::core::types::model::Usage;
    use crate::core::types::tools::ToolRun;
    use crate::core::types::trace::RunStatus;

    let answer = Answer {
        text: "Open 7am to 2am.".into(),
        evidence: vec![Evidence {
            source_id: uuid::Uuid::new_v4(),
            chunk_id: uuid::Uuid::new_v4(),
            key: "library_hours".into(),
            title: "Library hours".into(),
            content: String::new(),
            url: Some("https://lib.asu.edu/hours".into()),
            fetched_at: chrono::Utc::now(),
            score: 1.0,
        }],
        sources: vec![Citation {
            key: "courses".into(),
            title: "ASU Class Search".into(),
            url: Some("https://catalog.apps.asu.edu/catalog/classes/classlist?term=2267".into()),
        }],
        confirmation: None,
        status: RunStatus::Answered,
        steps: 2,
        tool_runs: vec![ToolRun {
            tool: "search_courses".into(),
            ok: true,
        }],
        memories: Vec::new(),
        usage: Usage::default(),
        cost_usd: 0.0,
    };

    let out = transcript(&answer);

    assert!(out.starts_with("Open 7am to 2am."));
    assert!(out.contains("search_courses"), "{out}");
    assert!(out.contains("lib.asu.edu/hours"), "{out}");
    assert!(
        out.contains("catalog.apps.asu.edu"),
        "a page a tool read is cited with the evidence: {out}"
    );
}

#[test]
fn a_bare_answer_carries_no_footers() {
    use crate::core::types::agent::Answer;
    use crate::core::types::model::Usage;
    use crate::core::types::trace::RunStatus;

    let answer = Answer {
        text: "No idea.".into(),
        evidence: Vec::new(),
        sources: Vec::new(),
        confirmation: None,
        status: RunStatus::Answered,
        steps: 1,
        tool_runs: Vec::new(),
        memories: Vec::new(),
        usage: Usage::default(),
        cost_usd: 0.0,
    };

    assert_eq!(transcript(&answer), "No idea.");
}
