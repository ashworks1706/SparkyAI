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
