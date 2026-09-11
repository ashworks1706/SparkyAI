//! The per request state the loop carries: what it loaded before the first model call, and
//! what it accumulates across steps.

use std::collections::HashSet;
use std::time::Instant;

use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::memory::Memory;
use crate::core::types::message::Message;
use crate::core::types::model::Usage;
use crate::core::types::tool::ToolRun;

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
    /// Evidence the tools found, in the order they found it.
    pub(super) tool_evidence: Vec<Evidence>,
    /// Set after a step of nothing but repeats. The next model call gets no tools.
    pub(super) force_answer: bool,
    /// Leading new_turns entries that assembly appends itself, which the prompt must not
    /// repeat. One for a fresh request, none when resuming after an approval.
    pub(super) appended_by_assembly: usize,
}

/// What an answer may cite: the chunks that fit the prompt, then what the tools found.
pub(super) fn cited(retrieved: Vec<Evidence>, run: &Run<'_>) -> Vec<Evidence> {
    let mut out: Vec<Evidence> = retrieved.into_iter().take(run.evidence_in_prompt).collect();
    for found in &run.tool_evidence {
        if !out.iter().any(|e| e.chunk_id == found.chunk_id) {
            out.push(found.clone());
        }
    }
    out
}
