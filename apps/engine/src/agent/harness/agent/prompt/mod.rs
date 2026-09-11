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
    /// Line naming the current date, with {date}.
    pub date_line: String,
    /// Hours from UTC the date is rendered in.
    pub utc_offset_hours: i32,
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
            date_line: cfg.date_line.clone(),
            utc_offset_hours: cfg.utc_offset_hours,
        }
    }
}

impl PromptText {
    /// Today's date in the configured offset, as the date line renders it.
    ///
    /// Config::validate bounds the offset, so it always resolves; UTC answers if it ever
    /// does not.
    pub fn today(&self) -> String {
        const FORMAT: &str = "%A %-d %B %Y";
        let now = chrono::Utc::now();
        match chrono::FixedOffset::east_opt(self.utc_offset_hours * 3600) {
            Some(offset) => now.with_timezone(&offset).format(FORMAT).to_string(),
            None => now.format(FORMAT).to_string(),
        }
    }

    /// Borrowed view for one assembly pass.
    pub(super) fn templates(&self) -> Templates<'_> {
        Templates {
            role_line: &self.role_line,
            role_line_no_roles: &self.role_line_no_roles,
            memory_header: &self.memory_header,
            evidence_header: &self.evidence_header,
            date_line: &self.date_line,
        }
    }
}
