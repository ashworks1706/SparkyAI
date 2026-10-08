//! Google Calendar tool: direct-message gating, connection gating, and the reply.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde_json::json;

use crate::core::traits::oauth::OAuthStore;
use crate::core::traits::tools::Tool;
use crate::core::traits::tools::gcal::GoogleCalendar;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::store::StoreError;
use crate::core::types::tools::ToolError;
use crate::core::types::tools::gcal::{GCalError, GCalEvent};
use crate::core::types::tools::oauth::{Consent, OAuthTokens};
use crate::runtime::tools::account::gcal::{GcalTool, tools};
use crate::runtime::tools::account::grant::Credentials;

/// A calendar double with canned events.
#[derive(Default)]
struct FakeGCal {
    events: Vec<GCalEvent>,
}

#[async_trait]
impl GoogleCalendar for FakeGCal {
    async fn events(&self, _token: &SecretString) -> Result<Vec<GCalEvent>, GCalError> {
        Ok(self.events.clone())
    }
}

/// A grant store double reporting no per-user grant.
struct NoGrants;

#[async_trait]
impl OAuthStore for NoGrants {
    async fn save_grant(
        &self,
        _t: &str,
        _u: &str,
        _p: &str,
        _tk: &OAuthTokens,
    ) -> Result<(), StoreError> {
        Ok(())
    }
    async fn load_grant(
        &self,
        _t: &str,
        _u: &str,
        _p: &str,
    ) -> Result<Option<OAuthTokens>, StoreError> {
        Ok(None)
    }
    async fn delete_grant(&self, _t: &str, _u: &str, _p: &str) -> Result<bool, StoreError> {
        Ok(false)
    }
    async fn begin_consent(
        &self,
        _s: &str,
        _t: &str,
        _u: &str,
        _p: &str,
        _ttl: Duration,
    ) -> Result<(), StoreError> {
        Ok(())
    }
    async fn take_consent(&self, _s: &str) -> Result<Option<Consent>, StoreError> {
        Ok(None)
    }
}

fn dm() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5)).with_visibility(Visibility::Private)
}

fn tool(fake: FakeGCal, connected: bool) -> GcalTool {
    let fallback = if connected { "tok" } else { "" };
    let creds = Credentials::new(
        Arc::new(NoGrants),
        None,
        "google",
        SecretString::from(fallback.to_owned()),
    );
    GcalTool::new(Arc::new(fake), Arc::new(creds), 15)
}

#[tokio::test]
async fn google_calendar_is_refused_outside_a_direct_message() {
    let server = RequestContext::new("g", "u", Duration::from_secs(5));
    let out = tool(FakeGCal::default(), true)
        .call(&server, json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("direct message")),
        "{out:?}"
    );
}

#[tokio::test]
async fn an_unconnected_user_is_told_to_log_in() {
    let out = tool(FakeGCal::default(), false)
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("/login")),
        "{out:?}"
    );
}

#[tokio::test]
async fn events_are_listed_with_summary_and_start() {
    let fake = FakeGCal {
        events: vec![GCalEvent {
            summary: "CSE 471 lecture".into(),
            start: Some("2026-10-05T14:00:00-07:00".into()),
            location: Some("COOR 170".into()),
            url: None,
        }],
    };
    let out = tool(fake, true).call(&dm(), json!({})).await;
    assert!(
        matches!(&out, Ok(o) if o.content.contains("CSE 471 lecture")
            && o.content.contains("COOR 170")),
        "{out:?}"
    );
}

#[test]
fn the_tool_builds_from_its_config() {
    let cfg = crate::core::config::Gcal::default();
    let built = tools(&cfg, Arc::new(NoGrants), None);
    assert!(built.is_ok_and(|tools| tools.len() == 1));
}
