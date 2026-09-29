//! Google Calendar tool: read the caller's upcoming events, in a direct message only.

pub mod client;

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde_json::{Value, json};

use crate::core::config::Gcal as GcalConfig;
use crate::core::traits::oauth::OAuthStore;
use crate::core::traits::tools::Tool;
use crate::core::traits::tools::gcal::GoogleCalendar;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::tools::gcal::{GCalError, GCalEvent};
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::tools::gcal::client::GCalClient;
use crate::runtime::tools::grant::Credentials;
use crate::runtime::tools::oauth::WebOAuthClient;
use crate::runtime::tools::structured;

/// The provider key Google grants are stored under.
const PROVIDER: &str = "google";

/// The Google Calendar read tool.
pub struct GcalTool {
    client: Arc<dyn GoogleCalendar>,
    creds: Arc<Credentials>,
    max_items: usize,
}

impl GcalTool {
    /// Builds the tool over the client and the shared credentials.
    pub fn new(client: Arc<dyn GoogleCalendar>, creds: Arc<Credentials>, max_items: usize) -> Self {
        Self {
            client,
            creds,
            max_items,
        }
    }
}

/// Maps a Calendar failure to the text the model reads.
fn failed(error: GCalError) -> ToolError {
    match error {
        GCalError::Refused(401 | 403) => {
            ToolError::Failed("Google rejected the token; it may have expired.".to_owned())
        }
        other => ToolError::Failed(other.to_string()),
    }
}

#[async_trait]
impl Tool for GcalTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "google_calendar".to_owned(),
            description: "List the user's upcoming Google Calendar events, soonest first. Works \
                          only in a direct message, and only after the user has connected Google."
                .to_owned(),
            parameters: json!({ "type": "object", "properties": {} }),
            risk: RiskClass::ReadAuthenticated,
            sequential: false,
            timeout_secs: None,
        }
    }

    async fn call(&self, ctx: &RequestContext, _args: Value) -> Result<ToolOutput, ToolError> {
        if ctx.visibility != Visibility::Private {
            return Err(ToolError::Failed(
                "Google Calendar is available only in a direct message with me, not in a server."
                    .to_owned(),
            ));
        }
        let Some(token) = self.creds.resolve(ctx).await? else {
            return Err(ToolError::Failed(
                "You have not connected Google. Send /login in a direct message to connect it."
                    .to_owned(),
            ));
        };
        let events = self.client.events(&token).await.map_err(failed)?;
        Ok(output(events, self.max_items))
    }
}

/// The calendar reply.
fn output(events: Vec<GCalEvent>, max: usize) -> ToolOutput {
    let shown: Vec<GCalEvent> = events.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No upcoming Google Calendar events.".to_owned()
    } else {
        let mut lines = format!("{} upcoming events:", shown.len());
        for e in &shown {
            let when = e.start.as_deref().unwrap_or("no start time");
            let place = e
                .location
                .as_deref()
                .map(|l| format!(" at {l}"))
                .unwrap_or_default();
            let link = e
                .url
                .as_deref()
                .map(|u| format!(" {u}"))
                .unwrap_or_default();
            let _ = write!(lines, "\n- {} {}{}{}", e.summary, when, place, link);
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}

/// The Google Calendar tool, over the client, the per-user grant store, and the refresh client.
pub fn tools(
    cfg: &GcalConfig,
    store: Arc<dyn OAuthStore>,
    oauth: Option<Arc<WebOAuthClient>>,
) -> Result<Vec<Arc<dyn Tool>>, GCalError> {
    let client: Arc<dyn GoogleCalendar> = Arc::new(GCalClient::new(
        &cfg.base_url,
        Duration::from_secs(cfg.timeout_secs),
        cfg.max_items,
    )?);
    let creds = Arc::new(Credentials::new(
        store,
        oauth,
        PROVIDER,
        SecretString::from(""),
    ));
    let tool: Arc<dyn Tool> = Arc::new(GcalTool::new(client, creds, cfg.max_items));
    Ok(vec![tool])
}
