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
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            total: 3_000,
            evidence: 1_200,
            history: 1_000,
            memory: 300,
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
