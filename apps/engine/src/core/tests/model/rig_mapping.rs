//! Mapping between core::types messages and the Rig message model.

use ::rig_core::completion::AssistantContent;
use ::rig_core::message::Message as RigMessage;
use serde_json::json;

use crate::core::types::conversation::message::{Message, ToolCall};
use crate::runtime::model::rig_openai::{from_rig, to_rig, with_thinking};

fn rig(messages: &[Message]) -> (Option<String>, Vec<RigMessage>) {
    to_rig(messages).unwrap_or_else(|e| unreachable!("{e}"))
}

#[test]
fn system_messages_become_the_preamble() {
    let (preamble, history) = rig(&[
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
    let (_, history) = rig(&[
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
fn response_content_splits_text_reasoning_and_calls() {
    let (text, reasoning, calls) = from_rig(vec![
        AssistantContent::text("hello"),
        AssistantContent::tool_call("c1", "echo", json!({"x": 2})),
        AssistantContent::reasoning("the hours are not in what I was given"),
    ]);
    assert_eq!(text, "hello");
    assert_eq!(
        reasoning, "the hours are not in what I was given",
        "what the model reasoned is kept, not dropped"
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "c1");
    assert_eq!(calls[0].name, "echo");
    assert_eq!(calls[0].arguments, json!({"x": 2}));
}

#[test]
fn a_tool_result_without_ids_is_rejected() {
    let mut orphan = Message::tool_result("c", "echo", "ok");
    orphan.tool_call_id = None;
    assert!(to_rig(&[orphan]).is_err());
}

#[test]
fn each_call_sets_the_thinking_switch_and_keeps_other_template_arguments() {
    let params = json!({"top_p": 0.9, "chat_template_kwargs": {"custom": 1}});
    let on = with_thinking(&params, true);
    assert_eq!(on["chat_template_kwargs"]["enable_thinking"], true);
    assert_eq!(on["chat_template_kwargs"]["custom"], 1);
    assert_eq!(on["top_p"], 0.9);
    let off = with_thinking(&json!({}), false);
    assert_eq!(off["chat_template_kwargs"]["enable_thinking"], false);
}
