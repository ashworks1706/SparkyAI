//! Mapping between `core::types` messages and Rig's message model.

use ::rig_core::completion::AssistantContent;
use ::rig_core::message::Message as RigMessage;
use serde_json::json;

use crate::agent::model::rig_openai::{from_rig, to_rig};
use crate::core::types::message::{Message, ToolCall};

#[test]
fn system_messages_become_the_preamble() {
    let (preamble, history) = to_rig(&[
        Message::system("rules"),
        Message::system("evidence"),
        Message::user("hi"),
    ]);
    assert_eq!(preamble.as_deref(), Some("rules\n\nevidence"));
    assert_eq!(history.len(), 1);
    assert!(matches!(history[0], RigMessage::User { .. }));
}

#[test]
fn tool_calls_and_results_round_trip_their_ids() {
    let call = ToolCall {
        id: "call_9".into(),
        name: "echo".into(),
        arguments: json!({"a": 1}),
    };
    let (_, history) = to_rig(&[
        Message::assistant_tool_calls("", vec![call]),
        Message::tool_result("call_9", "echo", "ok"),
    ]);
    assert_eq!(history.len(), 2);
    let first_is_tool_call = matches!(
        &history[0],
        RigMessage::Assistant { content, .. }
            if matches!(content.first(), Some(AssistantContent::ToolCall(_)))
    );
    assert!(first_is_tool_call);
    assert!(matches!(history[1], RigMessage::User { .. }));
}

#[test]
fn response_content_splits_text_and_calls() {
    let (text, calls) = from_rig(vec![
        AssistantContent::text("hello"),
        AssistantContent::tool_call("c1", "echo", json!({"x": 2})),
        AssistantContent::reasoning("thinking"),
    ]);
    assert_eq!(text, "hello");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "c1");
    assert_eq!(calls[0].name, "echo");
    assert_eq!(calls[0].arguments, json!({"x": 2}));
}
