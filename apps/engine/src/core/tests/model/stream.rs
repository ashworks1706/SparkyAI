//! The Rig chat adapter streaming reasoning, text, a tool call, and usage as llama-server does.

use secrecy::SecretString;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc::unbounded_channel;

use crate::core::tests::support::ctx;
use crate::core::traits::model::ModelProvider;
use crate::core::types::conversation::message::Message;
use crate::core::types::model::{FinishReason, ModelDelta, ModelRequest};
use crate::runtime::model::rig_openai::{RigChat, client};

/// Serves one streamed completion made of these data lines, then closes.
async fn serve(lines: &'static [&'static str]) -> String {
    let Ok(listener) = tokio::net::TcpListener::bind("127.0.0.1:0").await else {
        unreachable!("a local port is free")
    };
    let Ok(address) = listener.local_addr() else {
        unreachable!("the listener has an address")
    };
    tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };
        let mut request = vec![0u8; 65_536];
        let _ = socket.read(&mut request).await;
        let mut body = String::new();
        for line in lines {
            body.push_str("data: ");
            body.push_str(line);
            body.push_str("\n\n");
        }
        let head =
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
        let _ = socket.write_all(head.as_bytes()).await;
        let _ = socket.write_all(body.as_bytes()).await;
        let _ = socket.shutdown().await;
    });
    format!("http://{address}/v1")
}

fn chat(base_url: &str) -> RigChat {
    let Ok(client) = client(base_url, &SecretString::from("key")) else {
        unreachable!("the client builds")
    };
    RigChat::new(client, "qwen", serde_json::json!({}))
}

fn request() -> ModelRequest {
    ModelRequest {
        messages: vec![Message::system("sys"), Message::user("when")],
        tools: Vec::new(),
        max_tokens: 64,
        temperature: 0.0,
        thinking: true,
    }
}

#[tokio::test]
async fn reasoning_and_text_arrive_as_deltas_and_the_whole_completion_is_returned() {
    let url = serve(&[
        r#"{"id":"c","model":"qwen","choices":[{"index":0,"delta":{"reasoning_content":"Row two "}}]}"#,
        r#"{"id":"c","model":"qwen","choices":[{"index":0,"delta":{"reasoning_content":"has it."}}]}"#,
        r#"{"id":"c","model":"qwen","choices":[{"index":0,"delta":{"content":"Open until "}}]}"#,
        r#"{"id":"c","model":"qwen","choices":[{"index":0,"delta":{"content":"2am."},"finish_reason":"stop"}]}"#,
        r#"{"id":"c","model":"qwen","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19}}"#,
        "[DONE]",
    ])
    .await;
    let (tx, mut rx) = unbounded_channel();
    let response = chat(&url).stream(&ctx(), request(), tx).await;
    let Ok(response) = response else {
        unreachable!("the stream completed, got {response:?}")
    };
    let mut deltas = Vec::new();
    while let Ok(delta) = rx.try_recv() {
        deltas.push(delta);
    }
    assert_eq!(
        deltas,
        vec![
            ModelDelta::Reasoning("Row two ".into()),
            ModelDelta::Reasoning("has it.".into()),
            ModelDelta::Text("Open until ".into()),
            ModelDelta::Text("2am.".into()),
        ]
    );
    assert_eq!(response.content, "Open until 2am.");
    assert_eq!(response.reasoning, "Row two has it.");
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.prompt_tokens, 12);
    assert_eq!(response.usage.completion_tokens, 7);
}

#[tokio::test]
async fn a_streamed_tool_call_comes_back_whole_and_a_length_stop_is_reported() {
    let url = serve(&[
        r#"{"id":"c","model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search_courses","arguments":"{\"term\":"}}]}}]}"#,
        r#"{"id":"c","model":"qwen","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Fall 2026\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        r#"{"id":"c","model":"qwen","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}}"#,
        "[DONE]",
    ])
    .await;
    let (tx, _rx) = unbounded_channel();
    let response = chat(&url).stream(&ctx(), request(), tx).await;
    let Ok(response) = response else {
        unreachable!("the stream completed, got {response:?}")
    };
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].name, "search_courses");
    assert_eq!(response.tool_calls[0].arguments["term"], "Fall 2026");
    assert_eq!(response.finish_reason, FinishReason::ToolCalls);

    let url = serve(&[
        r#"{"id":"c","model":"qwen","choices":[{"index":0,"delta":{"reasoning_content":"still going"},"finish_reason":"length"}]}"#,
        r#"{"id":"c","model":"qwen","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":64,"total_tokens":69}}"#,
        "[DONE]",
    ])
    .await;
    let (tx, _rx) = unbounded_channel();
    let response = chat(&url).stream(&ctx(), request(), tx).await;
    assert!(
        response.is_ok_and(|r| r.finish_reason == FinishReason::Length && r.content.is_empty()),
        "a completion that ran out of room says so"
    );
}
