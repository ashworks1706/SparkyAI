//! valley_metro: active Valley Metro vehicles from a GTFS-realtime JSON feed. Off until a feed URL
//! is set. The feed gives route ids and positions; resolving route ids to names needs the static
//! GTFS feed, which is a later enhancement.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::core::config::Transit as TransitConfig;
use crate::core::traits::tools::Tool;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::tools::structured;

/// Reports active Valley Metro vehicles from a realtime feed.
pub struct TransitTool {
    http: reqwest::Client,
    feed_url: SecretString,
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
            description: "Report how many Valley Metro (Phoenix) vehicles are running now, by \
                          route. Pass a route to count only that route."
                .to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "route": { "type": "string", "description": "A route id to filter to, optional." }
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
        from_json(&body, route.as_deref(), self.max_items).map_err(ToolError::Failed)
    }
}

/// Renders a transit reply from a raw JSON body. Err when the body is not the expected shape.
pub(crate) fn from_json(body: &str, route: Option<&str>, max: usize) -> Result<ToolOutput, String> {
    let feed: FeedMessage =
        serde_json::from_str(body).map_err(|e| format!("transit feed sent an odd reply: {e}"))?;
    Ok(render(&feed, route, max))
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

/// The reply: active vehicle counts by route, or for one route.
fn render(feed: &FeedMessage, route: Option<&str>, max: usize) -> ToolOutput {
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
        if route.is_some_and(|want| !route_id.eq_ignore_ascii_case(want)) {
            continue;
        }
        *by_route.entry(route_id).or_insert(0) += 1;
    }
    let total: u32 = by_route.values().sum();
    let mut ranked: Vec<(String, u32)> = by_route.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let shown: Vec<(String, u32)> = ranked.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        match route {
            Some(r) => format!("No Valley Metro vehicles running on route {r} now."),
            None => "No Valley Metro vehicles running now.".to_owned(),
        }
    } else {
        let mut lines = format!("{total} Valley Metro vehicles running now, by route:");
        for (route_id, count) in &shown {
            let _ = write!(lines, "\n- route {route_id}: {count}");
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
    let tool: Arc<dyn Tool> = Arc::new(TransitTool {
        http,
        feed_url: cfg.feed_url.clone(),
        max_items: cfg.max_items,
    });
    Ok(vec![tool])
}
