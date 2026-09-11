//! OTLP export reaches both PostHog paths with the project token.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, Uri};
use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
use secrecy::SecretString;

use crate::core::config::Telemetry;
use crate::core::telemetry::provider;

type Seen = Arc<Mutex<Vec<(String, String)>>>;

async fn record(State(seen): State<Seen>, uri: Uri, headers: HeaderMap) -> StatusCode {
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    if let Ok(mut seen) = seen.lock() {
        seen.push((uri.path().to_owned(), auth));
    }
    StatusCode::OK
}

/// Serves on a thread of its own and returns the bound address.
fn serve(seen: Seen) -> Option<std::net::SocketAddr> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return;
        };
        rt.block_on(async move {
            let Ok(listener) = tokio::net::TcpListener::bind("127.0.0.1:0").await else {
                return;
            };
            let _ = tx.send(listener.local_addr().ok());
            let app = Router::new().fallback(record).with_state(seen);
            let _ = axum::serve(listener, app).await;
        });
    });
    rx.recv_timeout(Duration::from_secs(5)).ok().flatten()
}

#[test]
fn a_span_reaches_both_paths_with_the_bearer_token() {
    let seen: Seen = Arc::default();
    let addr = serve(Arc::clone(&seen));
    assert!(addr.is_some());
    let Some(addr) = addr else {
        return;
    };
    let cfg = Telemetry {
        host: Some(format!("http://{addr}/")),
        project_token: SecretString::from("phc_test"),
        ..Telemetry::default()
    };
    let built = provider(&cfg, "engine-test", "test");
    assert!(built.as_ref().is_ok_and(Option::is_some), "{built:?}");
    let Ok(Some(provider)) = built else {
        return;
    };
    let mut span = provider.tracer("engine-test").start("probe");
    span.end();
    let _ = provider.force_flush();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut got = Vec::new();
    while Instant::now() < deadline {
        got = seen.lock().map(|s| s.clone()).unwrap_or_default();
        if got.len() >= 2 {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = provider.shutdown();
    let mut paths: Vec<&str> = got.iter().map(|(p, _)| p.as_str()).collect();
    paths.sort_unstable();
    assert_eq!(paths, ["/i/v0/ai/otel", "/i/v1/traces"]);
    assert!(
        got.iter().all(|(_, auth)| auth == "Bearer phc_test"),
        "{got:?}"
    );
}

#[test]
fn an_empty_token_or_host_turns_export_off() {
    let no_token = Telemetry::default();
    assert!(matches!(provider(&no_token, "e", "test"), Ok(None)));
    let no_host = Telemetry {
        host: Some(" ".into()),
        project_token: SecretString::from("phc_test"),
        ..Telemetry::default()
    };
    assert!(matches!(provider(&no_host, "e", "test"), Ok(None)));
}
