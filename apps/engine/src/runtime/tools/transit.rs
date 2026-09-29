//! valley_metro: active Valley Metro vehicles from a GTFS-realtime JSON feed. Off until a feed URL
//! is set. Route ids are resolved to names from the static GTFS routes.txt when transit.routes_url
//! is set; the routes are fetched once and cached.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::OnceCell;

use crate::core::config::Transit as TransitConfig;
use crate::core::traits::tools::Tool;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::tools::structured;

/// Reports active Valley Metro vehicles from a realtime feed.
pub struct TransitTool {
    http: reqwest::Client,
    feed_url: SecretString,
    routes_url: Option<String>,
    route_names: OnceCell<HashMap<String, String>>,
    max_items: usize,
}

/// A GTFS-realtime feed as JSON.
#[derive(Debug, Deserialize)]
struct FeedMessage {
    #[serde(default = "Vec::new")]
    entity: Vec<Entity>,
}

/// One realtime entity; only vehicle positions are read.
#[derive(Debug, Deserialize)]
struct Entity {
    #[serde(default)]
    vehicle: Option<VehiclePosition>,
}

/// A vehicle's realtime position and trip.
#[derive(Debug, Deserialize)]
struct VehiclePosition {
    #[serde(default)]
    trip: Option<Trip>,
}

/// The trip a vehicle is serving.
#[derive(Debug, Deserialize)]
struct Trip {
    // GTFS-realtime JSON uses routeId; some serializers keep the protobuf route_id.
    #[serde(default, alias = "routeId")]
    route_id: Option<String>,
}

#[async_trait]
impl Tool for TransitTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "valley_metro".to_owned(),
            description:
                "Report how many Valley Metro (Phoenix) buses and trains are running now, \
                          by route. Pass a route to count only that route."
                    .to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "route": { "type": "string", "description": "A route id or name to filter to, optional." }
                }
            }),
            risk: RiskClass::ReadPublic,
            sequential: false,
            timeout_secs: None,
        }
    }

    async fn call(&self, _ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        let route = args
            .get("route")
            .and_then(Value::as_str)
            .map(|r| r.trim().to_owned())
            .filter(|r| !r.is_empty());
        let response = self
            .http
            .get(self.feed_url.expose_secret())
            .send()
            .await
            .map_err(|e| ToolError::Failed(format!("transit feed unreachable: {}", kind(&e))))?;
        if !response.status().is_success() {
            return Err(ToolError::Failed(format!(
                "transit feed returned {}",
                response.status().as_u16()
            )));
        }
        let body = response
            .text()
            .await
            .map_err(|e| ToolError::Failed(format!("transit feed unreachable: {}", kind(&e))))?;
        let feed: FeedMessage = serde_json::from_str(&body)
            .map_err(|e| ToolError::Failed(format!("transit feed sent an odd reply: {e}")))?;
        let names = self
            .route_names
            .get_or_init(|| async {
                match &self.routes_url {
                    Some(url) => fetch_routes(&self.http, url).await,
                    None => HashMap::new(),
                }
            })
            .await;
        Ok(render(&feed, route.as_deref(), self.max_items, names))
    }
}

/// Renders a transit reply from a raw JSON body, with no route names. For tests.
#[cfg(test)]
pub(crate) fn from_json(body: &str, route: Option<&str>, max: usize) -> Result<ToolOutput, String> {
    let feed: FeedMessage =
        serde_json::from_str(body).map_err(|e| format!("transit feed sent an odd reply: {e}"))?;
    Ok(render(&feed, route, max, &HashMap::new()))
}

/// Renders a transit reply from a feed body and a routes.txt body. For tests.
#[cfg(test)]
pub(crate) fn from_json_with_routes(
    body: &str,
    route: Option<&str>,
    max: usize,
    routes_csv: &str,
) -> Result<ToolOutput, String> {
    let feed: FeedMessage =
        serde_json::from_str(body).map_err(|e| format!("transit feed sent an odd reply: {e}"))?;
    Ok(render(&feed, route, max, &parse_routes_csv(routes_csv)))
}

/// Names a reqwest failure without repeating the URL or its key.
fn kind(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        "timed out".to_owned()
    } else if error.is_connect() {
        "could not connect".to_owned()
    } else {
        "request failed".to_owned()
    }
}

/// Fetches routes.txt and maps route id to name. Any failure leaves the map empty; ids stand.
async fn fetch_routes(http: &reqwest::Client, url: &str) -> HashMap<String, String> {
    match http.get(url).send().await {
        Ok(response) if response.status().is_success() => match response.text().await {
            Ok(text) => parse_routes_csv(&text),
            Err(error) => {
                tracing::warn!(error = %kind(&error), "transit routes body unreadable");
                HashMap::new()
            }
        },
        Ok(response) => {
            tracing::warn!(
                status = response.status().as_u16(),
                "transit routes fetch failed"
            );
            HashMap::new()
        }
        Err(error) => {
            tracing::warn!(error = %kind(&error), "transit routes unreachable");
            HashMap::new()
        }
    }
}

/// Maps route id to the long name, else the short name, from a routes.txt body.
fn parse_routes_csv(text: &str) -> HashMap<String, String> {
    let mut lines = text.lines();
    let Some(header) = lines.next() else {
        return HashMap::new();
    };
    let cols: Vec<String> = split_csv(header)
        .into_iter()
        .map(|c| c.trim().to_lowercase())
        .collect();
    let at = |name: &str| cols.iter().position(|c| c == name);
    let Some(id_at) = at("route_id") else {
        return HashMap::new();
    };
    let short_at = at("route_short_name");
    let long_at = at("route_long_name");
    let mut map = HashMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = split_csv(line);
        let Some(id) = fields
            .get(id_at)
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        let pick = |index: Option<usize>| {
            index
                .and_then(|i| fields.get(i))
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
        };
        if let Some(name) = pick(long_at).or_else(|| pick(short_at)) {
            map.insert(id, name);
        }
    }
    map
}

/// Splits one CSV line into fields, honoring double-quoted fields and escaped quotes.
fn split_csv(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                current.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => fields.push(std::mem::take(&mut current)),
            _ => current.push(ch),
        }
    }
    fields.push(current);
    fields
}

/// The label for a route: its name and id, or just the id when no name is known.
fn label(route_id: &str, names: &HashMap<String, String>) -> String {
    match names.get(route_id) {
        Some(name) => format!("{name} (route {route_id})"),
        None => format!("route {route_id}"),
    }
}

/// The reply: active vehicle counts by route, or for one route.
fn render(
    feed: &FeedMessage,
    route: Option<&str>,
    max: usize,
    names: &HashMap<String, String>,
) -> ToolOutput {
    let mut by_route: BTreeMap<String, u32> = BTreeMap::new();
    for entity in &feed.entity {
        let Some(vehicle) = &entity.vehicle else {
            continue;
        };
        let route_id = vehicle
            .trip
            .as_ref()
            .and_then(|t| t.route_id.clone())
            .unwrap_or_else(|| "unknown".to_owned());
        if let Some(want) = route {
            let name = names.get(&route_id).map(String::as_str).unwrap_or_default();
            if !route_id.eq_ignore_ascii_case(want) && !name.eq_ignore_ascii_case(want) {
                continue;
            }
        }
        *by_route.entry(route_id).or_insert(0) += 1;
    }
    let total: u32 = by_route.values().sum();
    let mut ranked: Vec<(String, u32)> = by_route.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let shown: Vec<(String, u32)> = ranked.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        match route {
            Some(r) => format!("No Valley Metro vehicles running on {r} now."),
            None => "No Valley Metro vehicles running now.".to_owned(),
        }
    } else {
        let mut lines = format!("{total} Valley Metro vehicles running now, by route:");
        for (route_id, count) in &shown {
            let _ = write!(lines, "\n- {}: {count}", label(route_id, names));
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}

/// The Valley Metro tool, when it is enabled and has a feed URL.
pub fn tools(cfg: &TransitConfig) -> Result<Vec<Arc<dyn Tool>>, String> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(cfg.timeout_secs))
        .build()
        .map_err(|e| e.to_string())?;
    let routes_url = Some(cfg.routes_url.trim().to_owned()).filter(|u| !u.is_empty());
    let tool: Arc<dyn Tool> = Arc::new(TransitTool {
        http,
        feed_url: cfg.feed_url.clone(),
        routes_url,
        route_names: OnceCell::new(),
        max_items: cfg.max_items,
    });
    Ok(vec![tool])
}
