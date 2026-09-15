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

#[test]
fn an_attached_image_becomes_an_image_block_beside_the_text() {
    use ::rig_core::message::{DocumentSourceKind, ImageMediaType, UserContent};

    use crate::core::types::conversation::image::Attachment;

    let attached = Attachment::new("https://cdn/one.png", "image/png");
    assert!(attached.is_some(), "a png is an image a model is sent");
    let turn = Message::user_with_images("what is this?", attached.into_iter().collect());

    let (_, history) = rig(&[turn]);
    let RigMessage::User { content } = &history[0] else {
        unreachable!("a user turn")
    };
    assert_eq!(content.len(), 2, "the text and the image");
    assert!(matches!(content[0], UserContent::Text(_)));
    match &content[1] {
        UserContent::Image(image) => {
            assert_eq!(image.data, DocumentSourceKind::url("https://cdn/one.png"));
            assert_eq!(image.media_type, Some(ImageMediaType::PNG));
        }
        other => unreachable!("an image block, not {other:?}"),
    }
}

#[test]
fn a_turn_with_no_images_stays_one_text_block() {
    let (_, history) = rig(&[Message::user("hi")]);
    let RigMessage::User { content } = &history[0] else {
        unreachable!("a user turn")
    };
    assert_eq!(content.len(), 1);
}

#[test]
fn the_http_surface_drops_an_image_type_it_was_not_promised() {
    use crate::core::types::conversation::image::Attachment;

    // The edge filters; a caller of the HTTP API is not trusted to have done so.
    let claimed = vec![
        Attachment {
            url: "https://cdn/one.png".into(),
            media_type: "image/png".into(),
        },
        Attachment {
            url: "https://cdn/payload.svg".into(),
            media_type: "text/html".into(),
        },
    ];
    let accepted = Attachment::accepted(claimed, 4);
    assert_eq!(accepted.len(), 1);
    assert_eq!(accepted[0].url, "https://cdn/one.png");
}
