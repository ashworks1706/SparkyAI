//! Google Calendar client over reqwest. Read-only, one token per call.

use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;

use crate::core::traits::tools::gcal::GoogleCalendar;
use crate::core::types::tools::gcal::{GCalError, GCalEvent};
use crate::runtime::tools::http;

/// A Google Calendar endpoint reached over HTTPS.
pub struct GCalClient {
    http: Client,
    base_url: String,
    page_size: usize,
}

/// An events list response.
#[derive(Debug, Deserialize)]
struct EventsResponse {
    #[serde(default = "Vec::new")]
    items: Vec<RawEvent>,
}

/// One event as the API returns it.
#[derive(Debug, Deserialize)]
struct RawEvent {
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    start: Option<RawStart>,
    #[serde(default)]
    location: Option<String>,
    #[serde(default, rename = "htmlLink")]
    html_link: Option<String>,
}

/// The nested start of an event: a timed dateTime or an all-day date.
#[derive(Debug, Deserialize)]
struct RawStart {
    #[serde(default, rename = "dateTime")]
    date_time: Option<String>,
    #[serde(default)]
    date: Option<String>,
}

impl GCalClient {
    /// Builds the client over the Calendar base URL, request budget, and page size.
    pub fn new(base_url: &str, timeout: Duration, page_size: usize) -> Result<Self, GCalError> {
        let http = http::client(timeout)
            .map_err(|e| GCalError::Unreachable(http::failure(&e).to_owned()))?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_owned(),
            page_size,
        })
    }
}

#[async_trait]
impl GoogleCalendar for GCalClient {
    async fn events(&self, token: &SecretString) -> Result<Vec<GCalEvent>, GCalError> {
        let url = reqwest::Url::parse_with_params(
            &format!("{}/calendars/primary/events", self.base_url),
            &[
                ("timeMin", Utc::now().to_rfc3339()),
                ("singleEvents", "true".to_owned()),
                ("orderBy", "startTime".to_owned()),
                ("maxResults", self.page_size.to_string()),
            ],
        )
        .map_err(|e| GCalError::Unreachable(e.to_string()))?;
        let response = self
            .http
            .get(url)
            .bearer_auth(token.expose_secret())
            .send()
            .await
            .map_err(|e| GCalError::Unreachable(http::failure(&e).to_owned()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(GCalError::Refused(status.as_u16()));
        }
        let list: EventsResponse = response
            .json()
            .await
            .map_err(|e| GCalError::Malformed(http::failure(&e).to_owned()))?;
        Ok(list
            .items
            .into_iter()
            .map(|e| GCalEvent {
                summary: e.summary.unwrap_or_else(|| "(no title)".to_owned()),
                start: e.start.and_then(|s| s.date_time.or(s.date)),
                location: e.location,
                url: e.html_link,
            })
            .collect())
    }
}
