//! The capabilities section: what the model is told it can do, and how each entry is kinded.

use crate::agent::harness::capability::{Capability, Kind, from_definitions, kind_of, render};
use crate::core::types::tool::{RiskClass, ToolDefinition};

fn definition(name: &str, risk: RiskClass) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: "does a thing".into(),
        parameters: serde_json::json!({"type": "object"}),
        risk,
        sequential: false,
        timeout_secs: None,
    }
}

#[test]
fn a_remote_tool_is_kinded_by_the_server_it_came_from() {
    let mcp = vec!["browser_click".to_owned()];
    assert_eq!(kind_of("browser_click", &mcp), Kind::Mcp);
    // A built-in with a browser-looking name is still a built-in.
    assert_eq!(kind_of("browser_click", &[]), Kind::Tool);
    assert_eq!(kind_of("search_knowledge_base", &mcp), Kind::Tool);
    assert_eq!(kind_of("get_skill", &mcp), Kind::Skill);
    assert_eq!(kind_of("run_sandbox", &mcp), Kind::Sandbox);
}

#[test]
fn every_offered_tool_becomes_one_line_naming_its_kind() {
    let defs = vec![
        definition("search_knowledge_base", RiskClass::ReadPublic),
        definition("browser_click", RiskClass::ExternalWrite),
    ];
    let caps = from_definitions(&defs, &["browser_click".to_owned()]);
    let text = render(&caps);
    assert!(text.contains("search_knowledge_base (tool)"), "{text}");
    assert!(text.contains("browser_click (mcp)"), "{text}");
    assert_eq!(
        text.lines().count(),
        3,
        "a heading and one line per capability"
    );
}

#[test]
fn a_capability_that_needs_approval_says_so() {
    // The model choosing a write tool without knowing it will stop for approval wastes a step.
    let caps = from_definitions(&[definition("post", RiskClass::ExternalWrite)], &[]);
    assert!(render(&caps).contains("needs the user to approve"));

    let caps = from_definitions(&[definition("look", RiskClass::ReadPublic)], &[]);
    assert!(!render(&caps).contains("needs the user to approve"));
}

#[test]
fn nothing_offered_writes_no_section() {
    // A heading with nothing under it spends budget and tells the model nothing.
    assert!(render(&[]).is_empty());
}

#[test]
fn the_section_reaches_the_prompt_and_is_capped_by_its_budget() {
    use crate::agent::harness::assemble::assemble;
    use crate::core::tests::support::ctx;
    use crate::core::types::assemble::{Budget, Sections};

    let caps = vec![Capability {
        name: "search_knowledge_base".into(),
        kind: Kind::Tool,
        description: "search the index".into(),
        risk: RiskClass::ReadPublic,
    }];
    let text = render(&caps);

    let out = assemble(
        &ctx(),
        &Sections {
            system: "sys",
            capabilities: &text,
            input: "hi",
            ..Sections::default()
        },
        Budget::default(),
    );
    let joined: String = out.messages.iter().map(|m| m.content.clone()).collect();
    assert!(joined.contains("search_knowledge_base (tool)"), "{joined}");

    // A section that does not fit is left out whole rather than truncated into a half list.
    let tight = assemble(
        &ctx(),
        &Sections {
            system: "sys",
            capabilities: &text,
            input: "hi",
            ..Sections::default()
        },
        Budget {
            capabilities: 1,
            ..Budget::default()
        },
    );
    let joined: String = tight.messages.iter().map(|m| m.content.clone()).collect();
    assert!(!joined.contains("search_knowledge_base (tool)"), "{joined}");
}
