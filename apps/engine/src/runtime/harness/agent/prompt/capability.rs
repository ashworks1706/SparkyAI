//! The capabilities section of the prompt: one list of what the model may do, each entry
//! labelled with its kind.

use std::fmt::Write as _;

use crate::core::types::tools::{RiskClass, ToolDefinition};

/// How a capability is carried out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A built-in tool.
    Tool,
    /// A tool on a remote MCP server.
    Mcp,
    /// A saved procedure the model follows.
    Skill,
    /// A command in an isolated environment.
    Sandbox,
}

impl Kind {
    /// Name this kind is listed under.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::Mcp => "mcp",
            Self::Skill => "skill",
            Self::Sandbox => "sandbox",
        }
    }
}

/// One thing the model may do.
#[derive(Debug, Clone)]
pub struct Capability {
    /// Name the model calls.
    pub name: String,
    /// How it is carried out.
    pub kind: Kind,
    /// What it does.
    pub description: String,
    /// What it can do to the world.
    pub risk: RiskClass,
}

/// The kind of the tool named name, given the names of the MCP tools.
pub fn kind_of(name: &str, mcp_names: &[String]) -> Kind {
    if name == "get_skill" {
        return Kind::Skill;
    }
    if name == "run_sandbox" {
        return Kind::Sandbox;
    }
    if mcp_names.iter().any(|m| m == name) {
        return Kind::Mcp;
    }
    Kind::Tool
}

/// Builds the list from the tool definitions the model is offered.
pub fn from_definitions(definitions: &[ToolDefinition], mcp_names: &[String]) -> Vec<Capability> {
    definitions
        .iter()
        .map(|d| Capability {
            name: d.name.clone(),
            kind: kind_of(&d.name, mcp_names),
            description: d.description.clone(),
            risk: d.risk,
        })
        .collect()
}

/// Renders the list, or an empty string when nothing is offered. Assembly writes the heading.
pub fn render(capabilities: &[Capability]) -> String {
    if capabilities.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for c in capabilities {
        let confirmed = if c.risk >= RiskClass::ExternalWrite {
            " [needs the user to approve it first]"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "- {} ({}){}: {}",
            c.name,
            c.kind.as_str(),
            confirmed,
            c.description.trim()
        );
    }
    out
}
