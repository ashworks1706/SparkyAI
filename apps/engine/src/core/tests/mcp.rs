//! MCP tool risk mapping.

use serde_json::json;

use crate::agent::tools::mcp::{compact_schema, required_only, risk_for};
use crate::core::types::tool::RiskClass;

#[test]
fn reads_and_inspection_run_freely() {
    for name in [
        "browser_navigate",
        "browser_snapshot",
        "browser_take_screenshot",
        "browser_find",
    ] {
        assert_eq!(risk_for(name), RiskClass::ReadPublic, "{name}");
    }
}

#[test]
fn interactions_are_drafts() {
    for name in [
        "browser_click",
        "browser_type",
        "browser_fill_form",
        "browser_press_key",
    ] {
        assert_eq!(risk_for(name), RiskClass::PrepareWrite, "{name}");
    }
}

#[test]
fn submits_and_unknowns_need_confirmation() {
    assert_eq!(risk_for("browser_submit_form"), RiskClass::ExternalWrite);
    assert_eq!(risk_for("browser_route"), RiskClass::ExternalWrite);
}

#[test]
fn required_only_drops_optional_properties() {
    let schema = json!({
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "filename": {"type": "string"},
            "target": {"type": "string"}
        },
        "required": ["url"]
    });
    let out = required_only(schema);
    let props = out["properties"]
        .as_object()
        .map(|p| p.keys().cloned().collect::<Vec<_>>());
    assert_eq!(props, Some(vec!["url".to_owned()]));
}

#[test]
fn compact_schema_trims_descriptions_and_noise() {
    let schema = json!({
        "$schema": "x",
        "title": "T",
        "properties": {"a": {"description": "y".repeat(500), "default": 1}}
    });
    let out = compact_schema(schema);
    assert!(out.get("$schema").is_none());
    assert!(out.get("title").is_none());
    assert!(out["properties"]["a"].get("default").is_none());
    assert!(
        out["properties"]["a"]["description"]
            .as_str()
            .is_some_and(|d| d.len() <= 80)
    );
}
