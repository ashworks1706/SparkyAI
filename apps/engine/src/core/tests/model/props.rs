//! Reading the context one slot of a llama-server holds.

use secrecy::SecretString;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::runtime::model::props::slot_context;

/// Serves body once as a JSON response, then closes.
async fn serve(status: &'static str, body: &'static str) -> String {
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
        let mut request = vec![0u8; 8_192];
        let _ = socket.read(&mut request).await;
        let head = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        let _ = socket.write_all(head.as_bytes()).await;
        let _ = socket.write_all(body.as_bytes()).await;
    });
    format!("http://{address}/v1")
}

#[tokio::test]
async fn the_slot_context_comes_from_the_default_generation_settings() {
    let url = serve(
        "200 OK",
        r#"{"default_generation_settings":{"n_ctx":4096},"total_slots":2}"#,
    )
    .await;
    assert_eq!(slot_context(&url, &SecretString::from("")).await, Ok(4096));
}

#[tokio::test]
async fn a_server_that_reports_no_context_is_an_error_not_a_zero() {
    let url = serve("200 OK", r#"{"model":"something else"}"#).await;
    assert!(slot_context(&url, &SecretString::from("")).await.is_err());
    let url = serve("404 Not Found", "{}").await;
    assert!(slot_context(&url, &SecretString::from("")).await.is_err());
}
