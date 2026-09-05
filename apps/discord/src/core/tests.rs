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

#[test]
fn sse_frames_come_out_whole_even_when_the_bytes_arrive_split() {
    use crate::sse::drain_frames;

    let mut buf = String::new();

    // A frame split across two reads yields nothing until it is complete.
    buf.push_str("event: progress\ndata: {\"text\":\"sear");
    assert!(drain_frames(&mut buf).is_empty());

    buf.push_str("ching ASU pages\"}\n\nevent: done\ndata: {}\n\n");
    let frames = drain_frames(&mut buf);
    assert_eq!(
        frames,
        vec![
            (
                "progress".to_owned(),
                "{\"text\":\"searching ASU pages\"}".to_owned()
            ),
            ("done".to_owned(), "{}".to_owned()),
        ]
    );
    assert!(buf.is_empty(), "consumed frames leave the buffer clean");
}

#[test]
fn a_frame_without_an_event_name_still_carries_its_data() {
    use crate::sse::drain_frames;

    let mut buf = String::from("data: {\"a\":1}\n\n");
    assert_eq!(
        drain_frames(&mut buf),
        vec![(String::new(), "{\"a\":1}".to_owned())]
    );
}

#[test]
fn a_component_id_survives_the_round_trip_and_rejects_anything_else() {
    use uuid::Uuid;

    use crate::components::{Action, CustomId};

    let token = Uuid::new_v4();
    let convo = Uuid::new_v4();
    let id = CustomId::new(Action::Approve, token, convo);
    let wire = id.to_string();

    assert!(wire.starts_with("sparky:"), "{wire}");
    assert!(wire.len() <= 100, "Discord caps custom_id at 100 bytes");
    assert_eq!(CustomId::parse(&wire), Some(id));

    // Anything the bot did not mint is not ours to act on.
    assert_eq!(CustomId::parse("approve"), None);
    assert_eq!(CustomId::parse("other:approve:x:y"), None);
    assert_eq!(CustomId::parse("sparky:approve:not-a-uuid:x"), None);
    assert_eq!(CustomId::parse("sparky:launch:x:y"), None, "unknown action");
}

#[test]
fn a_confirmation_offers_the_two_answers_and_a_plain_answer_offers_none() {
    use uuid::Uuid;

    use crate::components::rows_for;
    use crate::core::types::Confirmation;

    let resp = response("", vec![], "awaiting_confirmation");
    assert!(rows_for(&resp).is_empty(), "no confirmation, no buttons");

    let mut asked = response("", vec![], "awaiting_confirmation");
    asked.confirmation = Some(Confirmation {
        token: Uuid::new_v4(),
        tool: "browser_click".into(),
        summary: "Run browser_click.".into(),
    });
    let rows = rows_for(&asked);
    assert_eq!(rows.len(), 1, "one row of answers");
    assert_eq!(rows[0].len(), 2, "approve and deny");
}
