//! `RiskPolicy` decisions and confirmation payload hashing.

use std::time::Duration;

use serde_json::json;

use crate::agent::harness::policy::{Policy, RiskPolicy, payload_hash};
use crate::core::types::context::RequestContext;
use crate::core::types::policy::{Decision, ProposedAction};
use crate::core::types::tool::RiskClass;

fn ctx(roles: &[&str]) -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5))
        .with_roles(roles.iter().map(ToString::to_string).collect())
}

fn action(risk: RiskClass) -> ProposedAction {
    ProposedAction {
        tool: "t".into(),
        risk,
        arguments: json!({"a": 1}),
    }
}

#[tokio::test]
async fn reads_are_allowed() {
    let p = RiskPolicy::new(Some("Moderator".into()));
    let d = p.authorize(&ctx(&[]), &action(RiskClass::ReadPublic)).await;
    assert!(matches!(d, Ok(Decision::Allow)));
}

#[tokio::test]
async fn writes_without_role_are_denied() {
    let p = RiskPolicy::new(Some("Moderator".into()));
    let d = p
        .authorize(&ctx(&[]), &action(RiskClass::ExternalWrite))
        .await;
    assert!(matches!(d, Ok(Decision::Deny { .. })));
}

#[tokio::test]
async fn writes_with_role_need_confirmation() {
    let p = RiskPolicy::new(Some("Moderator".into()));
    let d = p
        .authorize(&ctx(&["Moderator"]), &action(RiskClass::ExternalWrite))
        .await;
    assert!(matches!(d, Ok(Decision::Confirm(_))));
}

#[tokio::test]
async fn forbidden_is_denied_regardless_of_role() {
    let p = RiskPolicy::new(None);
    let d = p
        .authorize(&ctx(&["Moderator"]), &action(RiskClass::Forbidden))
        .await;
    assert!(matches!(d, Ok(Decision::Deny { .. })));
}

#[test]
fn payload_hash_changes_with_arguments() {
    assert_ne!(
        payload_hash(&json!({"a": 1})),
        payload_hash(&json!({"a": 2}))
    );
    assert_eq!(
        payload_hash(&json!({"a": 1})),
        payload_hash(&json!({"a": 1}))
    );
}
