//! `RequestContext`: identity, deadline, cancellation.

use std::time::Duration;

use crate::core::types::context::RequestContext;

#[test]
fn contexts_are_distinct_per_request() {
    let a = RequestContext::new("g", "u1", Duration::from_secs(1));
    let b = RequestContext::new("g", "u1", Duration::from_secs(1));
    assert_ne!(a.request_id, b.request_id);
}

#[test]
fn expired_deadline_is_done() {
    let ctx = RequestContext::new("g", "u1", Duration::ZERO);
    assert!(ctx.is_done());
}

#[test]
fn cancel_is_done() {
    let ctx = RequestContext::new("g", "u1", Duration::from_mins(1));
    assert!(!ctx.is_done());
    ctx.cancel.cancel();
    assert!(ctx.is_done());
}

#[test]
fn roles_are_checked_by_name() {
    let ctx =
        RequestContext::new("g", "u1", Duration::from_secs(1)).with_roles(vec!["Moderator".into()]);
    assert!(ctx.has_role("Moderator"));
    assert!(!ctx.has_role("Admin"));
}
