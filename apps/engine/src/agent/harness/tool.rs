//! `ToolSet`: the tools available to one agent, keyed by name.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::core::traits::tool::Tool;
use crate::core::types::tool::ToolDefinition;

/// The tools available to one agent, keyed by name.
#[derive(Default, Clone)]
pub struct ToolSet {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a tool, replacing any with the same name.
    pub fn with(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.insert(tool.definition().name, tool);
        self
    }

    /// Looks a tool up by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Definitions in name order, for the model.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }
}

impl std::fmt::Debug for ToolSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.tools.keys()).finish()
    }
}
