//! Linked accounts on the platform: tokens and login links from the accounts module.

use secrecy::ExposeSecret;
use serde_json::json;

use super::{Fake, client};
use crate::core::traits::oauth::OAuthStore;
use crate::core::types::tools::oauth::USER_SCOPE;
use crate::stores::platform::PlatformAccounts;

const ACCOUNTS: &str = "/api/accounts/members/111";

#[tokio::test]
async fn a_connected_account_yields_its_access_token() {
    let fake = Fake::default();
    fake.on(
        "GET",
        &format!("{ACCOUNTS}/canvas/token"),
        200,
        json!({"access_token": "tok", "scopes": ["url:GET|/api/v1/courses"], "expires_at": "2026-10-08T03:00:00"}),
    );
    let store = PlatformAccounts::new(client(&fake).await);
    let Ok(Some(grant)) = store.load_grant(USER_SCOPE, "111", "canvas").await else {
        unreachable!("the grant loads")
    };
    assert_eq!(grant.access_token.expose_secret(), "tok");
    assert!(grant.refresh_token.is_none());
    assert_eq!(grant.scopes, vec!["url:GET|/api/v1/courses"]);
    assert!(grant.expires_at.is_some());
}

#[tokio::test]
async fn an_unconnected_or_expired_account_is_none() {
    let fake = Fake::default();
    fake.on(
        "GET",
        &format!("{ACCOUNTS}/canvas/token"),
        404,
        json!({"error": "The member has not connected canvas"}),
    );
    fake.on(
        "GET",
        &format!("{ACCOUNTS}/google/token"),
        409,
        json!({"error": "The google connection expired. The member has to connect again."}),
    );
    fake.on(
        "GET",
        &format!("{ACCOUNTS}/microsoft/token"),
        502,
        json!({"error": "microsoft could not refresh the token"}),
    );
    let store = PlatformAccounts::new(client(&fake).await);
    assert!(matches!(
        store.load_grant(USER_SCOPE, "111", "canvas").await,
        Ok(None)
    ));
    assert!(matches!(
        store.load_grant(USER_SCOPE, "111", "google").await,
        Ok(None)
    ));
    assert!(
        store
            .load_grant(USER_SCOPE, "111", "microsoft")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn the_platform_runs_the_login_and_hands_back_its_link() {
    let fake = Fake::default();
    fake.on(
        "POST",
        &format!("{ACCOUNTS}/canvas/login"),
        201,
        json!({"url": "https://platform.test/api/accounts/start/abc", "expires_at": "2026-10-08T01:10:00"}),
    );
    fake.on(
        "POST",
        &format!("{ACCOUNTS}/google/login"),
        404,
        json!({"error": "google accounts are not enabled on this platform"}),
    );
    let store = PlatformAccounts::new(client(&fake).await);
    assert!(store.hosts_login());
    assert!(matches!(
        store.login_link("111", "canvas").await,
        Ok(Some(url)) if url == "https://platform.test/api/accounts/start/abc"
    ));
    assert!(matches!(store.login_link("111", "google").await, Ok(None)));
    assert!(
        store
            .take_consent("anything")
            .await
            .is_ok_and(|c| c.is_none())
    );
}

#[tokio::test]
async fn disconnecting_removes_the_platform_grant() {
    let fake = Fake::default();
    fake.on(
        "DELETE",
        &format!("{ACCOUNTS}/canvas"),
        200,
        json!({"removed": true}),
    );
    let store = PlatformAccounts::new(client(&fake).await);
    assert!(matches!(
        store.delete_grant(USER_SCOPE, "111", "canvas").await,
        Ok(true)
    ));
}
