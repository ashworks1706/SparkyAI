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
fn filling_a_page_in_is_a_draft() {
    for name in [
        "browser_type",
        "browser_fill_form",
        "browser_select_option",
        "browser_hover",
    ] {
        assert_eq!(risk_for(name), RiskClass::PrepareWrite, "{name}");
    }
}

#[test]
fn anything_that_can_commit_the_page_needs_confirmation() {
    // Playwright MCP ships no tool with "submit" in its name: a form is submitted by clicking
    // a button or pressing Enter, and browser_evaluate runs arbitrary script in the page.
    for name in [
        "browser_click",
        "browser_press_key",
        "browser_evaluate",
        "browser_file_upload",
        "browser_handle_dialog",
    ] {
        assert_eq!(risk_for(name), RiskClass::ExternalWrite, "{name}");
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

/// Live smoke test against a running Playwright MCP server. Start it with `just browser`, then
/// `cargo test -p engine -- --ignored mcp_server`.
#[tokio::test]
#[ignore = "needs `just browser`"]
async fn mcp_server_lists_the_tools_the_engine_asks_for() {
    use crate::agent::tools::mcp;
    use crate::core::config::Mcp;

    let defaults = Mcp::default();
    let tools = mcp::connect(
        "http://127.0.0.1:8931/mcp",
        &defaults.playwright_tools,
        defaults.required_props_only,
    )
    .await
    .unwrap_or_default();
    assert!(!tools.is_empty(), "no tools; is `just browser` running?");

    let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
    for wanted in &defaults.playwright_tools {
        assert!(names.contains(wanted), "{wanted} missing from {names:?}");
    }
    let Some(navigate) = tools
        .iter()
        .find(|t| t.definition().name == "browser_navigate")
    else {
        unreachable!("browser_navigate is in the allowlist and was found above")
    };
    let schema = navigate.definition().parameters.to_string();
    assert!(schema.contains("url"), "{schema}");

    let output = navigate
        .call(
            &crate::core::tests::support::ctx(),
            serde_json::json!({"url": "about:blank"}),
        )
        .await
        .map(|o| o.content)
        .unwrap_or_default();
    assert!(!output.is_empty(), "browser_navigate returned nothing");
}
