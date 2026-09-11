//! RiskPolicy decisions and confirmation payload hashing.

use std::time::Duration;

use serde_json::json;

use crate::agent::harness::safety::policy::{RiskPolicy, payload_hash};
use crate::core::traits::safety::policy::Policy;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::safety::policy::{Decision, ProposedAction};
use crate::core::types::tools::RiskClass;

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
    let p = RiskPolicy::default();
    let d = p.authorize(&ctx(&[]), &action(RiskClass::ReadPublic)).await;
    assert!(matches!(d, Decision::Allow));
}

#[tokio::test]
async fn writes_without_role_are_denied() {
    let p = RiskPolicy::default();
    let d = p
        .authorize(&ctx(&[]), &action(RiskClass::ExternalWrite))
        .await;
    assert!(matches!(d, Decision::Deny { .. }));
}

#[tokio::test]
async fn writes_with_role_need_confirmation() {
    let p = RiskPolicy::default();
    let d = p
        .authorize(&ctx(&["MANAGE_GUILD"]), &action(RiskClass::ExternalWrite))
        .await;
    assert!(matches!(d, Decision::Confirm(_)));
}

#[tokio::test]
async fn forbidden_is_denied_regardless_of_role() {
    let p = RiskPolicy::default();
    let d = p
        .authorize(&ctx(&["MANAGE_GUILD"]), &action(RiskClass::Forbidden))
        .await;
    assert!(matches!(d, Decision::Deny { .. }));
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

#[tokio::test]
async fn the_role_that_may_write_is_configuration() {
    let p = RiskPolicy::new(
        vec!["officers".into()],
        false,
        crate::core::types::tools::RiskClass::ExternalWrite,
    );
    assert!(matches!(
        p.authorize(&ctx(&["MANAGE_GUILD"]), &action(RiskClass::ExternalWrite))
            .await,
        Decision::Deny { .. }
    ));
    assert!(matches!(
        p.authorize(&ctx(&["officers"]), &action(RiskClass::ExternalWrite))
            .await,
        Decision::Confirm(_)
    ));
}

#[tokio::test]
async fn an_empty_write_role_list_denies_everyone() {
    let p = RiskPolicy::new(Vec::new(), false, RiskClass::ExternalWrite);
    let d = p
        .authorize(&ctx(&["MANAGE_GUILD"]), &action(RiskClass::ExternalWrite))
        .await;
    assert!(matches!(d, Decision::Deny { .. }), "{d:?}");
}

#[tokio::test]
async fn confirm_from_moves_where_the_loop_stops_to_ask() {
    // Lowering it holds drafts too.
    let cautious = RiskPolicy::new(vec!["MANAGE_GUILD".into()], false, RiskClass::PrepareWrite);
    assert!(matches!(
        cautious
            .authorize(&ctx(&[]), &action(RiskClass::PrepareWrite))
            .await,
        Decision::Confirm(_)
    ));

    // Raising it past every class an unprivileged caller can reach lets drafts run.
    let relaxed = RiskPolicy::new(vec!["MANAGE_GUILD".into()], false, RiskClass::Destructive);
    assert!(matches!(
        relaxed
            .authorize(&ctx(&["MANAGE_GUILD"]), &action(RiskClass::ExternalWrite))
            .await,
        Decision::Allow
    ));
}

#[tokio::test]
async fn authenticated_reads_open_only_when_they_are_switched_on() {
    let closed = RiskPolicy::default();
    assert!(matches!(
        closed
            .authorize(&ctx(&[]), &action(RiskClass::ReadAuthenticated))
            .await,
        Decision::Deny { .. }
    ));
    let open = RiskPolicy::new(vec!["MANAGE_GUILD".into()], true, RiskClass::ExternalWrite);
    assert!(matches!(
        open.authorize(&ctx(&[]), &action(RiskClass::ReadAuthenticated))
            .await,
        Decision::Allow
    ));
}
