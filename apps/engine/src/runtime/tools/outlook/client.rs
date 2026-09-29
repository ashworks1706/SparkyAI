//! Microsoft Graph client over reqwest. Read-only, one token per call.

use std::time::Duration;

use async_trait::async_trait;
use chrono::{Duration as ChronoDuration, Utc};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;

use crate::core::traits::tools::outlook::Outlook;
use crate::core::types::tools::outlook::{OutlookError, OutlookEvent, OutlookMessage};

/// Names a reqwest failure without repeating the URL or the token.
fn kind_of(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        "timed out".to_owned()
    } else if error.is_connect() {
        "could not connect".to_owned()
    } else {
        "request failed".to_owned()
    }
}

/// A Microsoft Graph endpoint reached over HTTPS.
pub struct GraphClient {
    http: Client,
    base_url: String,
    page_size: usize,
}

/// A Graph collection response.
#[derive(Debug, Deserialize)]
struct GraphList<T> {
    #[serde(default = "Vec::new")]
    value: Vec<T>,
}

/// A calendar event as Graph returns it.
#[derive(Debug, Deserialize)]
struct RawEvent {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    start: Option<RawDateTime>,
    #[serde(default)]
    location: Option<RawLocation>,
    #[serde(default, rename = "webLink")]
    web_link: Option<String>,
}

/// The nested start of a Graph event.
#[derive(Debug, Deserialize)]
struct RawDateTime {
    #[serde(default, rename = "dateTime")]
    date_time: Option<String>,
}

/// The nested location of a Graph event.
#[derive(Debug, Deserialize)]
struct RawLocation {
    #[serde(default, rename = "displayName")]
    display_name: Option<String>,
}

/// A mail message as Graph returns it.
#[derive(Debug, Deserialize)]
struct RawMessage {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    from: Option<RawFrom>,
    #[serde(default, rename = "receivedDateTime")]
    received: Option<String>,
    #[serde(default, rename = "bodyPreview")]
    body_preview: Option<String>,
    #[serde(default, rename = "webLink")]
    web_link: Option<String>,
}

/// The sender of a Graph message.
#[derive(Debug, Deserialize)]
struct RawFrom {
    #[serde(default, rename = "emailAddress")]
    email_address: Option<RawAddress>,
}

/// One email address on Graph.
#[derive(Debug, Deserialize)]
struct RawAddress {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    address: Option<String>,
}

impl GraphClient {
    /// Builds the client over the Graph base URL, request budget, and page size.
    pub fn new(base_url: &str, timeout: Duration, page_size: usize) -> Result<Self, OutlookError> {
        let http = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| OutlookError::Unreachable(kind_of(&e)))?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_owned(),
            page_size,
        })
    }

    /// One authenticated GET of a Graph path, deserialized into a collection of T.
    async fn list<T: for<'de> Deserialize<'de>>(
        &self,
        token: &SecretString,
        path: &str,
        query: &[(&str, String)],
    ) -> Result<Vec<T>, OutlookError> {
        let url = reqwest::Url::parse_with_params(
            &format!("{}/{path}", self.base_url),
            query.iter().map(|(k, v)| (*k, v.as_str())),
        )
        .map_err(|e| OutlookError::Unreachable(e.to_string()))?;
        let response = self
            .http
            .get(url)
            .bearer_auth(token.expose_secret())
            // Graph returns event times in this zone when asked.
            .header("Prefer", "outlook.timezone=\"UTC\"")
            .send()
            .await
            .map_err(|e| OutlookError::Unreachable(kind_of(&e)))?;
        let status = response.status();
        if !status.is_success() {
            return Err(OutlookError::Refused(status.as_u16()));
        }
        let list: GraphList<T> = response
            .json()
            .await
            .map_err(|e| OutlookError::Malformed(kind_of(&e)))?;
        Ok(list.value)
    }
}

#[async_trait]
impl Outlook for GraphClient {
    async fn calendar(&self, token: &SecretString) -> Result<Vec<OutlookEvent>, OutlookError> {
        let now = Utc::now();
        let end = now + ChronoDuration::days(30);
        let raw: Vec<RawEvent> = self
            .list(
                token,
                "me/calendarView",
                &[
                    ("startDateTime", now.to_rfc3339()),
                    ("endDateTime", end.to_rfc3339()),
                    ("$orderby", "start/dateTime".to_owned()),
                    ("$top", self.page_size.to_string()),
                    ("$select", "subject,start,location,webLink".to_owned()),
                ],
            )
            .await?;
        Ok(raw
            .into_iter()
            .map(|e| OutlookEvent {
                subject: e.subject,
                start: e.start.and_then(|s| s.date_time),
                location: e.location.and_then(|l| l.display_name),
                url: e.web_link,
            })
            .collect())
    }

    async fn mail(&self, token: &SecretString) -> Result<Vec<OutlookMessage>, OutlookError> {
        let raw: Vec<RawMessage> = self
            .list(
                token,
                "me/messages",
                &[
                    ("$top", self.page_size.to_string()),
                    ("$orderby", "receivedDateTime desc".to_owned()),
                    (
                        "$select",
                        "subject,from,receivedDateTime,bodyPreview,webLink".to_owned(),
                    ),
                ],
            )
            .await?;
        Ok(raw
            .into_iter()
            .map(|m| {
                let from =
                    m.from
                        .and_then(|f| f.email_address)
                        .map(|a| match (a.name, a.address) {
                            (Some(name), Some(addr)) => format!("{name} <{addr}>"),
                            (Some(name), None) => name,
                            (None, Some(addr)) => addr,
                            (None, None) => String::new(),
                        });
                OutlookMessage {
                    subject: m.subject,
                    from: from.filter(|f| !f.is_empty()),
                    received: m.received,
                    preview: m.body_preview.filter(|p| !p.trim().is_empty()),
                    url: m.web_link,
                }
            })
            .collect())
    }
}
