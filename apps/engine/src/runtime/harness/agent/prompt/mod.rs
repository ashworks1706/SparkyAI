//! The configurable wording assembly writes around the prompt sections.

pub mod assemble;
pub mod capability;

use crate::core::types::agent::assemble::Templates;

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
    /// Line written when retrieval found nothing.
    pub no_evidence_line: String,
    /// Line written when the router skipped retrieval as small talk.
    pub no_retrieval_line: String,
    /// Line written when the router skipped retrieval because the answer has to be current.
    pub live_only_line: String,
    /// Heading above what the model may do.
    pub capabilities_header: String,
    /// Line naming the current date, with {date}.
    pub date_line: String,
    /// Line added to a call that is offered no tools.
    pub answer_only_line: String,
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
            no_evidence_line: cfg.no_evidence_line.clone(),
            no_retrieval_line: cfg.no_retrieval_line.clone(),
            live_only_line: cfg.live_only_line.clone(),
            capabilities_header: cfg.capabilities_header.clone(),
            date_line: cfg.date_line.clone(),
            answer_only_line: cfg.answer_only_line.clone(),
            utc_offset_hours: cfg.utc_offset_hours,
        }
    }
}

impl PromptText {
    /// Today's date in the configured offset, as the date line renders it.
    pub fn today(&self) -> String {
        const FORMAT: &str = "%A %-d %B %Y";
        let now = chrono::Utc::now();
        let offset = self
            .utc_offset_hours
            .checked_mul(3600)
            .and_then(chrono::FixedOffset::east_opt);
        if let Some(offset) = offset {
            return now.with_timezone(&offset).format(FORMAT).to_string();
        }
        tracing::warn!(
            utc_offset_hours = self.utc_offset_hours,
            "utc offset out of range; the date is rendered in UTC"
        );
        now.format(FORMAT).to_string()
    }

    /// Borrowed view for one assembly pass.
    pub(super) fn templates(&self) -> Templates<'_> {
        Templates {
            role_line: &self.role_line,
            role_line_no_roles: &self.role_line_no_roles,
            memory_header: &self.memory_header,
            evidence_header: &self.evidence_header,
            no_evidence_line: &self.no_evidence_line,
            no_retrieval_line: &self.no_retrieval_line,
            live_only_line: &self.live_only_line,
            capabilities_header: &self.capabilities_header,
            date_line: &self.date_line,
        }
    }
}
