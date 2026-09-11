//! The configurable wording assembly writes around the prompt sections.

use crate::core::types::assemble::Templates;

/// The configurable wording assembly writes around the sections.
#[derive(Debug, Clone)]
pub struct PromptText {
    /// Line naming the user, with {user} and {roles}.
    pub role_line: String,
    /// Line naming a user who holds no roles, with {user}.
    pub role_line_no_roles: String,
    /// Heading above recalled memories.
    pub memory_header: String,
    /// Heading above retrieved evidence.
    pub evidence_header: String,
}

impl Default for PromptText {
    fn default() -> Self {
        Self::from(&crate::core::config::Prompt::default())
    }
}

impl From<&crate::core::config::Prompt> for PromptText {
    fn from(cfg: &crate::core::config::Prompt) -> Self {
        Self {
            role_line: cfg.role_line.clone(),
            role_line_no_roles: cfg.role_line_no_roles.clone(),
            memory_header: cfg.memory_header.clone(),
            evidence_header: cfg.evidence_header.clone(),
        }
    }
}

impl PromptText {
    /// Borrowed view for one assembly pass.
    pub(super) fn templates(&self) -> Templates<'_> {
        Templates {
            role_line: &self.role_line,
            role_line_no_roles: &self.role_line_no_roles,
            memory_header: &self.memory_header,
            evidence_header: &self.evidence_header,
        }
    }
}
