//! Per-request state the loop carries: what loaded before the first call, what steps accumulate.

use std::collections::HashSet;
use std::time::Instant;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::message::Message;
use crate::core::types::knowledge::evidence::{Citation, Evidence};
use crate::core::types::knowledge::route::Route;
use crate::core::types::memory::Memory;
use crate::core::types::model::Usage;
use crate::core::types::tools::ToolRun;

/// What one request loaded before its first model call.
pub(super) struct Inputs {
    pub(super) history: Vec<Message>,
    pub(super) memory: Vec<Memory>,
    pub(super) evidence: Vec<Evidence>,
    /// What the router decided about retrieval for this request.
    pub(super) route: Route,
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
    /// Every tool and arguments pair already executed this request.
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
    /// Set once the run has been sent back to try the sandbox. It is offered once a turn.
    pub(super) sent_to_sandbox: bool,
    /// Leading new_turns entries that assembly appends itself, which the prompt must not repeat.
    pub(super) appended_by_assembly: usize,
}

impl<'a> Run<'a> {
    /// Fresh state for one request; new_turns holds the turns known before the first step.
    pub(super) fn new(
        ctx: &'a RequestContext,
        input: &'a str,
        new_turns: Vec<Message>,
        appended_by_assembly: usize,
    ) -> Self {
        Self {
            ctx,
            input,
            started: Instant::now(),
            steps: 0,
            usage: Usage::default(),
            new_turns,
            seen_calls: HashSet::new(),
            tool_runs: Vec::new(),
            evidence_in_prompt: 0,
            memories_in_prompt: Vec::new(),
            tool_sources: Vec::new(),
            force_answer: false,
            sent_to_sandbox: false,
            appended_by_assembly,
        }
    }
}

/// What an answer may cite from retrieval: the chunks that fit the prompt.
pub(super) fn cited(retrieved: Vec<Evidence>, run: &Run<'_>) -> Vec<Evidence> {
    retrieved.into_iter().take(run.evidence_in_prompt).collect()
}
