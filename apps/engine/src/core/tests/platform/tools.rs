//! Tool registration by mode: platform mode leaves Canvas and unpublished sources to the platform.

use std::collections::HashMap;
use std::sync::Arc;

use figment::Figment;
use figment::providers::{Format, Toml};
use serde_json::Value;

use crate::core::config::Config;
use crate::core::tests::support::{FakeQueries, HostedLogins, Stored};
use crate::core::types::knowledge::query::QuerySourceInfo;
use crate::core::types::tools::ToolDefinition;
use crate::wiring::tools::build_tools;

/// The sections with no defaults, with the local Canvas tools turned on.
const BASE: &str = r#"
[app]
env = "test"
http_addr = "127.0.0.1:0"
log_level = "info"
[engine]
service_token = "t"
[discord]
guild_id = 1
[model]
base_url = "http://localhost:8000/v1"
api_key = ""
name = "m"
max_tokens = 256
[embedding]
base_url = "http://localhost:8001/v1"
api_key = ""
name = "e"
dim = 1024
[canvas]
enabled = true
access_token = "canvas-test"
"#;

/// Reads the config without Config::validate, which refuses Canvas with platform.enabled.
fn config(platform: bool) -> Config {
    let mode = format!(
        "[platform]\nenabled = {platform}\nurl = \"https://platform.example.org\"\ntoken = \"plat_x\"\n"
    );
    match Figment::new()
        .merge(Toml::string(BASE))
        .merge(Toml::string(&mode))
        .extract()
    {
        Ok(cfg) => cfg,
        Err(e) => unreachable!("the test config parses: {e}"),
    }
}

/// A registry that publishes these keys.
fn registry(keys: &[&str]) -> Vec<QuerySourceInfo> {
    keys.iter()
        .map(|key| QuerySourceInfo {
            key: (*key).to_owned(),
            params: Vec::new(),
            indexed: false,
        })
        .collect()
}

/// Every tool definition the engine registers for cfg over a registry of keys.
async fn registered(cfg: &Config, keys: &[&str]) -> Vec<ToolDefinition> {
    let built = build_tools(
        cfg,
        Arc::new(FakeQueries::new(registry(keys))),
        Arc::new(Stored::empty()),
        None,
        Arc::new(HostedLogins),
        HashMap::new(),
    )
    .await;
    match built {
        Ok((tools, _)) => tools.definitions(),
        Err(e) => unreachable!("the tools build: {e}"),
    }
}

/// The source keys search_live offers.
fn live_sources(definitions: &[ToolDefinition]) -> Vec<String> {
    definitions
        .iter()
        .find(|d| d.name == "search_live")
        .and_then(|d| d.parameters.pointer("/properties/source/enum").cloned())
        .and_then(|keys| match keys {
            Value::Array(keys) => Some(
                keys.iter()
                    .filter_map(|k| k.as_str().map(str::to_owned))
                    .collect(),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn platform_mode_registers_no_local_canvas_tools() {
    let names: Vec<String> = registered(&config(true), &["web"])
        .await
        .into_iter()
        .map(|d| d.name)
        .collect();
    assert!(!names.iter().any(|n| n.starts_with("canvas_")), "{names:?}");
}

#[cfg(feature = "standalone")]
#[tokio::test]
async fn standalone_mode_registers_the_local_canvas_tools() {
    let names: Vec<String> = registered(&config(false), &["web"])
        .await
        .into_iter()
        .map(|d| d.name)
        .collect();
    assert!(names.iter().any(|n| n == "canvas_grades"), "{names:?}");
}

#[tokio::test]
async fn platform_mode_offers_only_the_sources_the_platform_publishes() {
    let offered = live_sources(&registered(&config(true), &["events", "dining"]).await);
    assert!(!offered.iter().any(|k| k == "clubs"), "{offered:?}");
    assert!(offered.iter().any(|k| k == "events"), "{offered:?}");
    // The fallback stays offered, so a call that names no source still has one.
    assert!(offered.iter().any(|k| k == "web"), "{offered:?}");
}

#[cfg(feature = "standalone")]
#[tokio::test]
async fn standalone_mode_offers_every_source_the_engine_knows() {
    let offered = live_sources(&registered(&config(false), &["events"]).await);
    assert!(offered.iter().any(|k| k == "clubs"), "{offered:?}");
}
