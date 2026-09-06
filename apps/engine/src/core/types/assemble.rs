//! Budget, Sections, and Assembled are the inputs and output of context assembly.

use crate::core::types::evidence::Evidence;
use crate::core::types::memory::Memory;
use crate::core::types::message::Message;

/// Budgets for one assembled prompt, in estimated tokens. Default is implemented in
/// core::config.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Whole prompt, everything included.
    pub total: usize,
    /// Cap on the evidence section.
    pub evidence: usize,
    /// Cap on prior turns.
    pub history: usize,
    /// Cap on the memory section.
    pub memory: usize,
    /// Characters per token the estimator assumes. Never zero.
    pub chars_per_token: usize,
}

/// The text assembly writes around the sections. Every field is configurable.
#[derive(Debug, Clone, Copy)]
pub struct Templates<'a> {
    /// Line naming the user, with {user} and {roles}.
    pub role_line: &'a str,
    /// Line naming a user who holds no roles, with {user}.
    pub role_line_no_roles: &'a str,
    /// Heading above recalled memories.
    pub memory_header: &'a str,
    /// Heading above retrieved evidence.
    pub evidence_header: &'a str,
}

/// Default line naming the user, with {user} and {roles}.
pub const ROLE_LINE: &str = "The user is `{user}`. Roles: {roles}.";
/// Default line naming a user who holds no roles, with {user}.
pub const ROLE_LINE_NO_ROLES: &str = "The user is `{user}`. They hold no special roles.";
/// Default heading above recalled memories.
pub const MEMORY_HEADER: &str = "What you remember about this user:";
/// Default heading above retrieved evidence.
pub const EVIDENCE_HEADER: &str = "Evidence from ASU sources. Answer only from this; cite \
                                   sources by number. If it does not answer the question, say so.";

impl Default for Templates<'_> {
    fn default() -> Self {
        Self {
            role_line: ROLE_LINE,
            role_line_no_roles: ROLE_LINE_NO_ROLES,
            memory_header: MEMORY_HEADER,
            evidence_header: EVIDENCE_HEADER,
        }
    }
}

/// Everything that can go into a prompt.
#[derive(Debug, Default)]
pub struct Sections<'a> {
    /// Versioned system instructions.
    pub system: &'a str,
    /// Recalled memories, best first.
    pub memory: &'a [Memory],
    /// Retrieved evidence, best first.
    pub evidence: &'a [Evidence],
    /// Prior turns and the tool exchanges of this request, oldest first.
    pub history: &'a [Message],
    /// The current message from the user.
    pub input: &'a str,
    /// Wording written around the sections.
    pub templates: Templates<'a>,
}

/// The assembled prompt plus what was left out.
#[derive(Debug)]
pub struct Assembled {
    /// Messages to send, in order.
    pub messages: Vec<Message>,
    /// Rough token estimate of the whole thing.
    pub estimated_tokens: usize,
    /// Evidence chunks that made it in.
    pub evidence_used: usize,
}
