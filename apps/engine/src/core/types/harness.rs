//! Structs the harness is built from: the agent and its dependencies, loop state, the tool
//! registry, the default policy, and the trace sinks. Behaviour lives in `agent::harness`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::core::traits::conversation::ConversationStore;
use crate::core::traits::memory::MemoryStore;
use crate::core::traits::model::ModelProvider;
use crate::core::traits::policy::Policy;
use crate::core::traits::retrieval::Retriever;
use crate::core::traits::tool::Tool;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::AgentConfig;
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::memory::Memory;
use crate::core::types::message::Message;
use crate::core::types::model::Usage;
use crate::core::types::policy::ConfirmationRequest;
use crate::core::types::trace::{RunStatus, TraceRecord};

/// The dependencies the loop drives. Every one is a trait with a test double.
pub struct AgentDeps {
    /// The chat model.
    pub model: Arc<dyn ModelProvider>,
    /// Tools the model may call.
    pub tools: ToolSet,
    /// Gates every tool call.
    pub policy: Arc<dyn Policy>,
    /// Receives every event.
    pub trace: Arc<dyn TraceSink>,
    /// Evidence, when configured.
    pub retriever: Option<Arc<dyn Retriever>>,
    /// Conversation history, when configured.
    pub conversations: Option<Arc<dyn ConversationStore>>,
    /// Cross-conversation memory, when configured.
    pub memory: Option<Arc<dyn MemoryStore>>,
}

/// The loop. Cheap to clone; holds only `Arc`s.
#[derive(Clone)]
pub struct Agent {
    pub(crate) deps: Arc<AgentDeps>,
    pub(crate) cfg: AgentConfig,
    pub(crate) system_prompt: Arc<str>,
}

/// What one request loaded before its first model call.
pub struct Inputs {
    pub(crate) history: Vec<Message>,
    pub(crate) memory: Vec<Memory>,
    pub(crate) evidence: Vec<Evidence>,
}

/// Mutable state carried across steps.
pub struct Run<'a> {
    pub(crate) ctx: &'a RequestContext,
    pub(crate) input: &'a str,
    pub(crate) started: Instant,
    pub(crate) steps: u32,
    pub(crate) usage: Usage,
    /// Turns produced during this request, persisted at the end. First is the user input.
    pub(crate) new_turns: Vec<Message>,
}

/// What a step decided.
pub enum StepOutcome {
    /// Keep looping.
    Continue,
    /// Stop with this status and text.
    Stop(RunStatus, String, Option<ConfirmationRequest>),
}

/// The tools available to one agent, keyed by name.
#[derive(Default, Clone)]
pub struct ToolSet {
    pub(crate) tools: BTreeMap<String, Arc<dyn Tool>>,
}

/// The default policy: risk class alone decides. Roles gate writes.
#[derive(Debug, Clone)]
pub struct RiskPolicy {
    /// Role required for `ExternalWrite` and `Destructive`.
    pub write_role: Option<String>,
}

/// Appends one JSON line per event to `<dir>/<request_id>.jsonl`.
#[derive(Debug)]
pub struct JsonlSink {
    pub(crate) dir: PathBuf,
}

/// Keeps every record in memory. For tests and for the admin trace viewer.
#[derive(Debug, Default)]
pub struct MemorySink {
    pub(crate) records: Mutex<Vec<TraceRecord>>,
}

/// Fans one event out to several sinks.
pub struct MultiSink(pub Vec<Arc<dyn TraceSink>>);
