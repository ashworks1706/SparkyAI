//! The sandbox routes: who may read them, and what a kill accepts.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use secrecy::SecretString;
use tower::ServiceExt as _;

use crate::routes::sandbox::SandboxState;

const TOKEN: &str = "change-me";

fn router() -> axum::Router {
    axum::Router::new()
        .route(
            "/sandbox",
            axum::routing::get(crate::routes::sandbox::report),
        )
        .route(
            "/sandbox/enabled",
            axum::routing::post(crate::routes::sandbox::switch),
        )
        .route(
            "/sandbox/{name}",
            axum::routing::delete(crate::routes::sandbox::kill),
        )
        .with_state(SandboxState {
            sandbox: Some(std::sync::Arc::new(
                crate::runtime::tools::sandbox::ContainerSandbox::default(),
            )),
            service_token: SecretString::from(TOKEN),
        })
}

async fn status(request: Request<Body>) -> StatusCode {
    match router().oneshot(request).await {
        Ok(response) => response.status(),
        Err(e) => unreachable!("the router answered with {e}"),
    }
}

fn get(path: &str, token: Option<&str>) -> Request<Body> {
    build("GET", path, token, Body::empty())
}

fn build(method: &str, path: &str, token: Option<&str>, body: Body) -> Request<Body> {
    let mut request = Request::builder().method(method).uri(path);
    if let Some(token) = token {
        request = request.header("authorization", format!("Bearer {token}"));
    }
    match request
        .header("content-type", "application/json")
        .body(body)
    {
        Ok(request) => request,
        Err(e) => unreachable!("the request could not be built: {e}"),
    }
}

#[tokio::test]
async fn every_sandbox_route_wants_the_bearer_token() {
    let switch = || Body::from(r#"{"enabled":false}"#);
    for (method, path, body) in [
        ("GET", "/sandbox", Body::empty()),
        ("POST", "/sandbox/enabled", switch()),
        ("DELETE", "/sandbox/anything", Body::empty()),
    ] {
        assert_eq!(
            status(build(method, path, None, body)).await,
            StatusCode::UNAUTHORIZED,
            "{method} {path} answered without a token"
        );
    }
    for (method, path, body) in [
        ("GET", "/sandbox", Body::empty()),
        ("POST", "/sandbox/enabled", switch()),
    ] {
        assert_eq!(
            status(build(method, path, Some("wrong"), body)).await,
            StatusCode::UNAUTHORIZED,
            "{method} {path} answered a wrong token"
        );
    }
}

#[tokio::test]
async fn a_body_is_not_parsed_before_the_token_is_checked() {
    let refused = status(build(
        "POST",
        "/sandbox/enabled",
        None,
        Body::from("not json at all"),
    ))
    .await;
    assert_eq!(
        refused,
        StatusCode::UNAUTHORIZED,
        "an unauthenticated caller never reaches the parser"
    );
}

#[tokio::test]
async fn a_container_the_engine_never_started_is_not_removed() {
    let refused = status(build(
        "DELETE",
        "/sandbox/some-other-container",
        Some(TOKEN),
        Body::empty(),
    ))
    .await;
    assert_eq!(
        refused,
        StatusCode::NOT_FOUND,
        "only a container this engine holds can be killed through the route"
    );
}

#[tokio::test]
async fn the_report_reads_with_the_token() {
    assert_eq!(status(get("/sandbox", Some(TOKEN))).await, StatusCode::OK);
}
