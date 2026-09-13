//! The per request state the loop carries: what it loaded before the first model call, and
//! what it accumulates across steps.

use std::collections::HashSet;
use std::time::Instant;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::message::Message;
use crate::core::types::knowledge::evidence::{Citation, Evidence};
use crate::core::types::memory::Memory;
use crate::core::types::model::Usage;
use crate::core::types::tools::ToolRun;

/// What one request loaded before its first model call.
pub(super) struct Inputs {
    pub(super) history: Vec<Message>,
    pub(super) memory: Vec<Memory>,
    pub(super) evidence: Vec<Evidence>,
}

/// Mutable state carried across steps.
pub(super) struct Run<'a> {
    pub(super) ctx: &'a RequestContext,
    pub(super) input: &'a str,
    pub(super) started: Instant,
    pub(super) steps: u32,
    pub(super) usage: Usage,
    /// Turns produced during this request, persisted at the end. First is the user input.
    pub(super) new_turns: Vec<Message>,
    /// Every (tool, arguments) already executed this request, to catch loops.
    pub(super) seen_calls: HashSet<String>,
    /// Tools that ran, in order, for the answer and the client.
    pub(super) tool_runs: Vec<ToolRun>,
    /// Evidence chunks that fit the prompt on the most recent step.
    pub(super) evidence_in_prompt: usize,
    /// Content of the memories that fit the prompt on the most recent step.
    pub(super) memories_in_prompt: Vec<String>,
    /// Pages the tools read, in the order they read them.
    pub(super) tool_sources: Vec<Citation>,
    /// Set after a step of nothing but repeats. The next model call gets no tools.
    pub(super) force_answer: bool,
    /// Leading new_turns entries that assembly appends itself, which the prompt must not
    /// repeat. One for a fresh request, none when resuming after an approval.
    pub(super) appended_by_assembly: usize,
}

/// What an answer may cite from retrieval: the chunks that fit the prompt.
pub(super) fn cited(retrieved: Vec<Evidence>, run: &Run<'_>) -> Vec<Evidence> {
    retrieved.into_iter().take(run.evidence_in_prompt).collect()
}
