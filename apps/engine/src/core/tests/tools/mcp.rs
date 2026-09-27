//! MCP tool risk mapping.

use serde_json::json;

use crate::core::types::tools::RiskClass;
use crate::runtime::tools::mcp::{compact_schema, required_only, risk_for};

#[test]
fn reads_and_inspection_run_freely() {
    for name in [
        "page_navigate",
        "page_snapshot",
        "take_screenshot",
        "find_text",
    ] {
        assert_eq!(risk_for(name), RiskClass::ReadPublic, "{name}");
    }
}

#[test]
fn filling_a_page_in_is_a_draft() {
    for name in ["type_text", "fill_form", "select_option", "hover"] {
        assert_eq!(risk_for(name), RiskClass::PrepareWrite, "{name}");
    }
}

#[test]
fn anything_that_can_commit_the_page_needs_confirmation() {
    // Clicks, key presses, uploads, dialogs, and script evaluation are external writes.
    for name in [
        "click",
        "press_key",
        "evaluate",
        "file_upload",
        "handle_dialog",
    ] {
        assert_eq!(risk_for(name), RiskClass::ExternalWrite, "{name}");
    }
}

#[test]
fn submits_and_unknowns_need_confirmation() {
    assert_eq!(risk_for("submit_form"), RiskClass::ExternalWrite);
    assert_eq!(risk_for("route"), RiskClass::ExternalWrite);
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
    let out = compact_schema(schema, 80);
    assert!(out.get("$schema").is_none());
    assert!(out.get("title").is_none());
    assert!(out["properties"]["a"].get("default").is_none());
    assert!(
        out["properties"]["a"]["description"]
            .as_str()
            .is_some_and(|d| d.len() <= 80)
    );
}

#[test]
fn a_pinned_risk_replaces_the_one_derived_from_the_name() {
    use std::collections::BTreeMap;

    use crate::runtime::tools::mcp::pinned_risk;

    let risks = BTreeMap::from([
        ("update_event".to_owned(), RiskClass::Destructive),
        ("list_events".to_owned(), RiskClass::ReadAuthenticated),
    ]);
    assert_eq!(pinned_risk("update_event", &risks), RiskClass::Destructive);
    assert_eq!(
        pinned_risk("list_events", &risks),
        RiskClass::ReadAuthenticated
    );
    assert_eq!(
        pinned_risk("create_event", &risks),
        RiskClass::ExternalWrite
    );
    assert_eq!(pinned_risk("page_snapshot", &risks), RiskClass::ReadPublic);
}

#[test]
fn a_pin_the_server_does_not_list_is_reported() {
    use std::collections::BTreeMap;

    use crate::runtime::tools::mcp::unoffered;

    let risks = BTreeMap::from([
        ("create_event".to_owned(), RiskClass::ExternalWrite),
        ("list_events".to_owned(), RiskClass::ReadAuthenticated),
    ]);
    let offered = vec!["list_events".to_owned()];
    assert_eq!(unoffered(&risks, &offered), vec!["create_event".to_owned()]);
    assert!(unoffered(&BTreeMap::new(), &offered).is_empty());
}
