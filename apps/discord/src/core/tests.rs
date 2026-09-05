//! Bot unit tests: reply chunking and rendering.

use uuid::Uuid;

use crate::bot::can_write;
use crate::core::types::ChatResponse;
use crate::reply::{MAX_MESSAGE, chunk, render};
use serenity::all::Permissions;

fn response(text: &str, citations: Vec<String>, status: &str) -> ChatResponse {
    ChatResponse {
        request_id: Uuid::new_v4(),
        conversation_id: Uuid::new_v4(),
        text: text.into(),
        citations,
        confirmation: None,
        status: status.into(),
        tools: Vec::new(),
    }
}

#[test]
fn short_text_is_one_message() {
    assert_eq!(chunk("hello", MAX_MESSAGE), vec!["hello".to_owned()]);
}

#[test]
fn long_text_splits_on_line_boundaries_under_the_limit() {
    let text = (0..200)
        .map(|i| format!("line {i} {}", "x".repeat(40)))
        .collect::<Vec<_>>()
        .join("\n");
    let parts = chunk(&text, 500);
    assert!(parts.len() > 1);
    assert!(parts.iter().all(|p| p.len() <= 500));
    assert!(parts.iter().all(|p| !p.ends_with('\n')));
    let rejoined = parts.join("\n");
    assert!(rejoined.contains("line 199"));
}

#[test]
fn citations_are_appended_as_a_footer() {
    let out = render(&response("2am", vec!["lib — url".into()], "answered"));
    assert_eq!(out.len(), 1);
    assert!(out[0].contains("**Sources**"));
    assert!(out[0].contains("1. lib — url"));
}

#[test]
fn empty_answer_explains_the_status() {
    let out = render(&response("", vec![], "deadline"));
    assert!(out[0].contains("too long"));
}

#[test]
fn no_traceparent_without_an_active_span() {
    use crate::engine_client::current_traceparent;

    assert!(current_traceparent().is_none());
}

#[test]
fn discord_management_permissions_grant_write_access() {
    assert!(can_write(Permissions::MANAGE_GUILD));
    assert!(can_write(Permissions::ADMINISTRATOR));
    assert!(!can_write(Permissions::MANAGE_MESSAGES));
}

#[test]
fn a_guild_role_cannot_impersonate_the_write_capability() {
    use crate::bot::{WRITE_CAPABILITY, authorized_roles};

    let named = vec!["students".to_owned(), WRITE_CAPABILITY.to_owned()];
    assert_eq!(
        authorized_roles(named.clone(), Some(Permissions::MANAGE_MESSAGES)),
        vec!["students".to_owned()]
    );
    assert_eq!(
        authorized_roles(named, Some(Permissions::MANAGE_GUILD)),
        vec!["students".to_owned(), WRITE_CAPABILITY.to_owned()]
    );
    assert_eq!(
        authorized_roles(vec!["students".to_owned()], None),
        vec!["students".to_owned()]
    );
}

#[test]
fn capacity_and_outage_read_differently_to_the_user() {
    use crate::core::types::EngineError;
    use crate::reply::failure;

    let busy = failure(&EngineError::Status {
        status: 503,
        body: "the model is at capacity".into(),
    });
    assert!(busy.contains("busy"), "{busy}");

    let down = failure(&EngineError::Transport("connection refused".into()));
    assert!(down.contains("unavailable"), "{down}");

    let broken = failure(&EngineError::Status {
        status: 502,
        body: String::new(),
    });
    assert!(broken.contains("unavailable"), "{broken}");
}

#[test]
fn tools_the_agent_ran_are_listed_under_the_answer() {
    use crate::core::types::ToolRun;

    let mut resp = response("Hayden closes at 2am.", vec![], "answered");
    resp.tools = vec![
        ToolRun {
            tool: "browser_navigate".into(),
            ok: true,
        },
        ToolRun {
            tool: "browser_snapshot".into(),
            ok: false,
        },
    ];

    let out = render(&resp).join("\n");

    assert!(out.contains("browser_navigate"), "{out}");
    assert!(out.contains("browser_snapshot"), "{out}");
    assert!(out.contains("failed"), "a failed call is marked: {out}");

    let quiet = render(&response("hi", vec![], "answered")).join("\n");
    assert!(!quiet.contains("Tools"), "no tools, no footer: {quiet}");
}
