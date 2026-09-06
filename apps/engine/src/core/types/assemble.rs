//! `Budget`, `Sections`, `Assembled` — inputs and output of context assembly.

use crate::core::types::evidence::Evidence;
use crate::core::types::memory::Memory;
use crate::core::types::message::Message;

/// Budgets for one assembled prompt, in estimated tokens.
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

impl Default for Budget {
    fn default() -> Self {
        Self {
            total: 3_000,
            evidence: 1_200,
            history: 1_000,
            memory: 300,
            chars_per_token: 4,
        }
    }
}

/// The text assembly writes around the sections. Every field is configurable, so the wording
/// of a prompt is a setting rather than a rebuild.
#[derive(Debug, Clone, Copy)]
pub struct Templates<'a> {
    /// Line naming the user, with `{user}` and `{roles}`.
    pub role_line: &'a str,
    /// Line naming a user who holds no roles, with `{user}`.
    pub role_line_no_roles: &'a str,
    /// Heading above recalled memories.
    pub memory_header: &'a str,
    /// Heading above retrieved evidence.
    pub evidence_header: &'a str,
}

impl Default for Templates<'_> {
    fn default() -> Self {
        Self {
            role_line: "The user is `{user}`. Roles: {roles}.",
            role_line_no_roles: "The user is `{user}`. They hold no special roles.",
            memory_header: "What you remember about this user:",
            evidence_header: "Evidence from ASU sources. Answer only from this; cite sources by \
                              number. If it does not answer the question, say so.",
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
    /// Prior turns and this request's own tool exchanges, oldest first.
    pub history: &'a [Message],
    /// The user's current message.
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
