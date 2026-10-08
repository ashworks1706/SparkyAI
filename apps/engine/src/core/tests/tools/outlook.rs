//! Outlook tools: direct-message gating, connection gating, and the replies.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde_json::json;

use crate::core::traits::oauth::OAuthStore;
use crate::core::traits::tools::Tool;
use crate::core::traits::tools::outlook::Outlook;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::store::StoreError;
use crate::core::types::tools::ToolError;
use crate::core::types::tools::oauth::{Consent, OAuthTokens};
use crate::core::types::tools::outlook::{OutlookError, OutlookEvent, OutlookMessage};
use crate::runtime::tools::account::grant::Credentials;
use crate::runtime::tools::account::outlook::{OutlookTool, Query};

/// An Outlook double with canned rows.
#[derive(Default)]
struct FakeOutlook {
    events: Vec<OutlookEvent>,
    mail: Vec<OutlookMessage>,
}

#[async_trait]
impl Outlook for FakeOutlook {
    async fn calendar(&self, _token: &SecretString) -> Result<Vec<OutlookEvent>, OutlookError> {
        Ok(self.events.clone())
    }
    async fn mail(&self, _token: &SecretString) -> Result<Vec<OutlookMessage>, OutlookError> {
        Ok(self.mail.clone())
    }
}

/// A grant store double that always reports a connection through the shared fallback.
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

fn tool(query: Query, fake: FakeOutlook, connected: bool) -> OutlookTool {
    let fallback = if connected { "tok" } else { "" };
    let creds = Credentials::new(
        Arc::new(NoGrants),
        None,
        "microsoft",
        SecretString::from(fallback.to_owned()),
    );
    OutlookTool::new(query, Arc::new(fake), Arc::new(creds), 15)
}

#[tokio::test]
async fn outlook_is_refused_outside_a_direct_message() {
    let server = RequestContext::new("g", "u", Duration::from_secs(5));
    let out = tool(Query::Calendar, FakeOutlook::default(), true)
        .call(&server, json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("direct message")),
        "{out:?}"
    );
}

#[tokio::test]
async fn an_unconnected_user_is_told_to_log_in() {
    let out = tool(Query::Mail, FakeOutlook::default(), false)
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("/login")),
        "{out:?}"
    );
}

#[tokio::test]
async fn calendar_lists_subject_and_location() {
    let fake = FakeOutlook {
        events: vec![OutlookEvent {
            subject: "Advising".into(),
            start: Some("2026-10-05T14:00:00".into()),
            location: Some("SSV 250".into()),
            url: None,
        }],
        ..FakeOutlook::default()
    };
    let out = tool(Query::Calendar, fake, true)
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Ok(o) if o.content.contains("Advising") && o.content.contains("SSV 250")),
        "{out:?}"
    );
}

#[tokio::test]
async fn mail_lists_subject_and_sender() {
    let fake = FakeOutlook {
        mail: vec![OutlookMessage {
            subject: "Registration opens".into(),
            from: Some("ASU Registrar <reg@asu.edu>".into()),
            received: Some("2026-09-29".into()),
            preview: Some("Your enrollment window...".into()),
            url: None,
        }],
        ..FakeOutlook::default()
    };
    let out = tool(Query::Mail, fake, true).call(&dm(), json!({})).await;
    assert!(
        matches!(&out, Ok(o) if o.content.contains("Registration opens")
            && o.content.contains("asu.edu")),
        "{out:?}"
    );
}
