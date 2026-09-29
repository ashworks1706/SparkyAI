//! The Google OAuth client: consent URL, code exchange, refresh, and failures.

use secrecy::{ExposeSecret, SecretString};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;

use crate::core::config::{CanvasOAuth, GoogleOAuth};
use crate::core::types::tools::oauth::{OAuthError, OAuthTokens};
use crate::runtime::tools::oauth::{CanvasOAuthClient, GoogleOAuthClient};

/// Serves one response and hands back the request it answered.
async fn serve(status: &'static str, body: &'static str) -> (String, oneshot::Receiver<String>) {
    let Ok(listener) = tokio::net::TcpListener::bind("127.0.0.1:0").await else {
        unreachable!("a local port is free")
    };
    let Ok(address) = listener.local_addr() else {
        unreachable!("the listener has an address")
    };
    let (tx, rx) = oneshot::channel();
    tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };
        let mut request = vec![0u8; 8_192];
        let n = socket.read(&mut request).await.unwrap_or(0);
        let _ = tx.send(String::from_utf8_lossy(&request[..n]).into_owned());
        let head = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        let _ = socket.write_all(head.as_bytes()).await;
        let _ = socket.write_all(body.as_bytes()).await;
    });
    (format!("http://{address}/token"), rx)
}

fn client(token_url: &str) -> GoogleOAuthClient {
    let cfg = GoogleOAuth {
        enabled: true,
        client_id: "cid".into(),
        client_secret: SecretString::from("csecret"),
        redirect_url: "https://sparky.example/oauth/google".into(),
        token_url: token_url.into(),
        ..GoogleOAuth::default()
    };
    match GoogleOAuthClient::new(&cfg) {
        Ok(c) => c,
        Err(e) => unreachable!("the client builds: {e}"),
    }
}

fn grant(refresh: Option<&str>) -> OAuthTokens {
    OAuthTokens {
        access_token: SecretString::from("old"),
        refresh_token: refresh.map(SecretString::from),
        scopes: vec!["https://www.googleapis.com/auth/calendar.events".into()],
        expires_at: None,
    }
}

#[test]
fn the_consent_url_asks_for_offline_access_with_the_state() {
    let url = client("https://oauth2.googleapis.com/token").authorize_url("s1");
    assert!(url.starts_with("https://accounts.google.com/o/oauth2/v2/auth?"));
    for part in [
        "client_id=cid",
        "response_type=code",
        "state=s1",
        "access_type=offline",
        "prompt=consent",
        "scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcalendar.events",
        "redirect_uri=https%3A%2F%2Fsparky.example%2Foauth%2Fgoogle",
    ] {
        assert!(url.contains(part), "{part} missing from {url}");
    }
    assert!(!url.contains("csecret"));
}

#[tokio::test]
async fn a_code_becomes_tokens_and_the_form_carries_the_client() {
    let (url, request) = serve(
        "200 OK",
        r#"{"access_token":"a1","refresh_token":"r1","scope":"s1 s2","expires_in":3600}"#,
    )
    .await;
    let tokens = match client(&url).exchange("code-1").await {
        Ok(t) => t,
        Err(e) => unreachable!("the exchange succeeds: {e}"),
    };
    assert_eq!(tokens.access_token.expose_secret(), "a1");
    assert_eq!(
        tokens
            .refresh_token
            .as_ref()
            .map(ExposeSecret::expose_secret),
        Some("r1")
    );
    assert_eq!(tokens.scopes, vec!["s1", "s2"]);
    assert!(tokens.expires_at.is_some());
    let sent = request.await.unwrap_or_default();
    for part in [
        "grant_type=authorization_code",
        "code=code-1",
        "client_id=cid",
        "client_secret=csecret",
    ] {
        assert!(sent.contains(part), "{part} missing from the token request");
    }
}

#[tokio::test]
async fn a_grant_without_a_refresh_token_is_refused() {
    let (url, _) = serve("200 OK", r#"{"access_token":"a1"}"#).await;
    assert_eq!(
        client(&url).exchange("c").await.err(),
        Some(OAuthError::NoRefreshToken)
    );
}

#[tokio::test]
async fn a_refresh_keeps_the_refresh_token_and_scopes_it_is_not_sent_again() {
    let (url, request) = serve("200 OK", r#"{"access_token":"a2","expires_in":3600}"#).await;
    let tokens = match client(&url).refresh(&grant(Some("r1"))).await {
        Ok(t) => t,
        Err(e) => unreachable!("the refresh succeeds: {e}"),
    };
    assert_eq!(tokens.access_token.expose_secret(), "a2");
    assert_eq!(
        tokens
            .refresh_token
            .as_ref()
            .map(ExposeSecret::expose_secret),
        Some("r1")
    );
    assert_eq!(tokens.scopes, grant(None).scopes);
    let sent = request.await.unwrap_or_default();
    assert!(sent.contains("grant_type=refresh_token"));
    assert!(sent.contains("refresh_token=r1"));
}

#[tokio::test]
async fn a_grant_with_no_refresh_token_cannot_be_refreshed() {
    let c = client("https://oauth2.googleapis.com/token");
    assert_eq!(
        c.refresh(&grant(None)).await.err(),
        Some(OAuthError::NoRefreshToken)
    );
}

#[tokio::test]
async fn a_refusal_names_the_status_and_code_but_not_the_body() {
    let (url, _) = serve(
        "400 Bad Request",
        r#"{"error":"invalid_grant","error_description":"Token has been revoked r1"}"#,
    )
    .await;
    let e = client(&url).refresh(&grant(Some("r1"))).await.err();
    assert_eq!(
        e,
        Some(OAuthError::Refused {
            status: 400,
            code: "invalid_grant".into()
        })
    );
    let text = e.map(|e| e.to_string()).unwrap_or_default();
    assert!(!text.contains("revoked"));
}

#[tokio::test]
async fn a_response_without_an_access_token_is_malformed() {
    let (url, _) = serve("200 OK", r#"{"refresh_token":"r1"}"#).await;
    assert!(matches!(
        client(&url).exchange("c").await.err(),
        Some(OAuthError::Malformed(_))
    ));
}

fn canvas_client(token_url: &str) -> CanvasOAuthClient {
    let cfg = CanvasOAuth {
        enabled: true,
        client_id: "cid".into(),
        client_secret: SecretString::from("csecret"),
        redirect_url: "https://sparky.example/oauth/canvas/callback".into(),
        token_url: token_url.into(),
        ..CanvasOAuth::default()
    };
    match CanvasOAuthClient::new(&cfg) {
        Ok(c) => c,
        Err(e) => unreachable!("the canvas client builds: {e}"),
    }
}

#[test]
fn the_canvas_consent_url_omits_offline_and_carries_the_state() {
    let url = canvas_client("https://canvas.asu.edu/login/oauth2/token").authorize_url("s9");
    assert!(url.starts_with("https://canvas.asu.edu/login/oauth2/auth?"));
    for part in [
        "client_id=cid",
        "response_type=code",
        "state=s9",
        "redirect_uri=https%3A%2F%2Fsparky.example%2Foauth%2Fcanvas%2Fcallback",
    ] {
        assert!(url.contains(part), "{part} missing from {url}");
    }
    assert!(!url.contains("access_type"));
    assert!(!url.contains("prompt=consent"));
}

#[tokio::test]
async fn a_canvas_code_becomes_tokens_without_requiring_a_refresh_token() {
    let (url, _) = serve("200 OK", r#"{"access_token":"a1","expires_in":3600}"#).await;
    let tokens = match canvas_client(&url).exchange("code-9").await {
        Ok(t) => t,
        Err(e) => unreachable!("the canvas exchange succeeds: {e}"),
    };
    assert_eq!(tokens.access_token.expose_secret(), "a1");
    assert!(tokens.refresh_token.is_none());
}
