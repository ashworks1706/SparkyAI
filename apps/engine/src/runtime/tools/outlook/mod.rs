//! Outlook tools: read the caller's calendar and mail, in a direct message only.

pub mod client;

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde_json::{Value, json};

use crate::core::config::Outlook as OutlookConfig;
use crate::core::traits::oauth::OAuthStore;
use crate::core::traits::tools::Tool;
use crate::core::traits::tools::outlook::Outlook;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::tools::outlook::{OutlookError, OutlookEvent, OutlookMessage};
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::tools::grant::Credentials;
use crate::runtime::tools::oauth::WebOAuthClient;
use crate::runtime::tools::outlook::client::GraphClient;
use crate::runtime::tools::structured;

/// The provider key Outlook grants are stored under.
const PROVIDER: &str = "microsoft";

/// Which read an Outlook tool performs.
#[derive(Debug, Clone, Copy)]
pub enum Query {
    /// Upcoming calendar events.
    Calendar,
    /// Recent mail.
    Mail,
}

/// Every read an Outlook tool is built for.
const ALL: [Query; 2] = [Query::Calendar, Query::Mail];

impl Query {
    /// The tool name the model calls.
    fn name(self) -> &'static str {
        match self {
            Self::Calendar => "outlook_calendar",
            Self::Mail => "outlook_mail",
        }
    }

    /// What the tool does, for the model.
    fn description(self) -> &'static str {
        match self {
            Self::Calendar => {
                "List the user's upcoming Outlook calendar events, soonest first. Works only in a \
                 direct message, and only after the user has connected Outlook."
            }
            Self::Mail => {
                "List the subjects and senders of the user's most recent Outlook mail. Works only \
                 in a direct message, and only after the user has connected Outlook."
            }
        }
    }
}

/// One Outlook read, offered to the model.
pub struct OutlookTool {
    query: Query,
    client: Arc<dyn Outlook>,
    creds: Arc<Credentials>,
    max_items: usize,
}

impl OutlookTool {
    /// Builds a tool over the client and the shared credentials.
    pub fn new(
        query: Query,
        client: Arc<dyn Outlook>,
        creds: Arc<Credentials>,
        max_items: usize,
    ) -> Self {
        Self {
            query,
            client,
            creds,
            max_items,
        }
    }
}

/// Maps a Graph failure to the text the model reads.
fn failed(error: OutlookError) -> ToolError {
    match error {
        OutlookError::Refused(401 | 403) => {
            ToolError::Failed("Outlook rejected the token; it may have expired.".to_owned())
        }
        other => ToolError::Failed(other.to_string()),
    }
}

#[async_trait]
impl Tool for OutlookTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.query.name().to_owned(),
            description: self.query.description().to_owned(),
            parameters: json!({ "type": "object", "properties": {} }),
            risk: RiskClass::ReadAuthenticated,
            sequential: false,
            timeout_secs: None,
        }
    }

    async fn call(&self, ctx: &RequestContext, _args: Value) -> Result<ToolOutput, ToolError> {
        if ctx.visibility != Visibility::Private {
            return Err(ToolError::Failed(
                "Outlook is available only in a direct message with me, not in a server."
                    .to_owned(),
            ));
        }
        let Some(token) = self.creds.resolve(ctx).await? else {
            return Err(ToolError::Failed(
                "You have not connected Outlook. Send /login in a direct message to connect it."
                    .to_owned(),
            ));
        };
        match self.query {
            Query::Calendar => {
                let events = self.client.calendar(&token).await.map_err(failed)?;
                Ok(calendar_output(events, self.max_items))
            }
            Query::Mail => {
                let mail = self.client.mail(&token).await.map_err(failed)?;
                Ok(mail_output(mail, self.max_items))
            }
        }
    }
}

/// The calendar reply.
fn calendar_output(events: Vec<OutlookEvent>, max: usize) -> ToolOutput {
    let shown: Vec<OutlookEvent> = events.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No upcoming Outlook events.".to_owned()
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
            let _ = write!(lines, "\n- {} {}{}{}", e.subject, when, place, link);
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}

/// The mail reply.
fn mail_output(mail: Vec<OutlookMessage>, max: usize) -> ToolOutput {
    let shown: Vec<OutlookMessage> = mail.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No recent Outlook mail.".to_owned()
    } else {
        let mut lines = format!("{} recent messages:", shown.len());
        for m in &shown {
            let from = m
                .from
                .as_deref()
                .map(|f| format!(" from {f}"))
                .unwrap_or_default();
            let when = m
                .received
                .as_deref()
                .map(|r| format!(" ({r})"))
                .unwrap_or_default();
            let preview = m
                .preview
                .as_deref()
                .map(|p| format!(" - {p}"))
                .unwrap_or_default();
            let _ = write!(lines, "\n- {}{}{}{}", m.subject, from, when, preview);
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}

/// The Outlook tools, over the Graph client, the per-user grant store, and the refresh client.
pub fn tools(
    cfg: &OutlookConfig,
    store: Arc<dyn OAuthStore>,
    oauth: Option<Arc<WebOAuthClient>>,
) -> Result<Vec<Arc<dyn Tool>>, OutlookError> {
    let client: Arc<dyn Outlook> = Arc::new(GraphClient::new(
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
    Ok(ALL
        .into_iter()
        .map(|q| {
            Arc::new(OutlookTool::new(
                q,
                client.clone(),
                creds.clone(),
                cfg.max_items,
            )) as Arc<dyn Tool>
        })
        .collect())
}
