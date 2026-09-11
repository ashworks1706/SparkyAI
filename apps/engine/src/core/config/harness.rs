//! Settings that shape the agent loop, the prompt, and what a tool is allowed to do.

use serde::Deserialize;

use crate::core::types::assemble::{self, Budget};
use crate::core::types::tool::RiskClass;

/// Agent loop limits. Every field has a default so a bare .env still boots.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Agent {
    /// Model calls per request.
    pub max_steps: u32,
    /// Retries per model call on transport or 5xx errors.
    pub max_model_retries: u32,
    /// Wall-clock budget per request.
    pub request_timeout_secs: u64,
    /// Budget per tool call, unless the tool declares its own.
    pub tool_timeout_secs: u64,
    /// How long a held action waits for caller approval.
    pub confirmation_ttl_secs: u64,
    /// Model calls in flight at once. Match llama-server --parallel. 0 removes the limit.
    pub model_slots: usize,
    /// How long a request waits for a free model slot before reporting the model busy.
    pub model_queue_wait_secs: u64,
    /// Sampling temperature.
    pub temperature: f32,
    /// Prior turns loaded into the prompt.
    pub history_turns: usize,
    /// Memories recalled per request.
    pub memory_recall_limit: usize,
    /// Whole-prompt token budget.
    pub prompt_budget_tokens: usize,
    /// Cap on the evidence section.
    pub evidence_budget_tokens: usize,
    /// Cap on prior turns.
    pub history_budget_tokens: usize,
    /// Cap on the memory section.
    pub memory_budget_tokens: usize,
    /// Cap on the capabilities section.
    pub capabilities_budget_tokens: usize,
    /// Characters per token the budget estimator assumes.
    pub chars_per_token: usize,
    /// First retry wait, doubled per attempt.
    pub retry_base_ms: u64,
    /// Longest retry wait.
    pub retry_cap_ms: u64,
    /// Longest value recorded on a span; the JSONL trace keeps the rest.
    pub max_span_value_chars: usize,
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_model_retries: 2,
            request_timeout_secs: 90,
            tool_timeout_secs: 20,
            confirmation_ttl_secs: 600,
            model_slots: 2,
            model_queue_wait_secs: 30,
            temperature: 0.3,
            history_turns: 20,
            memory_recall_limit: 10,
            prompt_budget_tokens: 3_000,
            evidence_budget_tokens: 1_200,
            history_budget_tokens: 1_000,
            memory_budget_tokens: 300,
            capabilities_budget_tokens: 600,
            chars_per_token: 4,
            retry_base_ms: 250,
            retry_cap_ms: 8_000,
            max_span_value_chars: 32_000,
        }
    }
}

impl Agent {
    /// The prompt budgets these settings describe.
    pub fn budget(&self) -> Budget {
        Budget {
            total: self.prompt_budget_tokens,
            evidence: self.evidence_budget_tokens,
            history: self.history_budget_tokens,
            memory: self.memory_budget_tokens,
            capabilities: self.capabilities_budget_tokens,
            chars_per_token: self.chars_per_token,
        }
    }
}

/// The text the harness writes around every prompt. Changing any of it changes the prompt hash.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Prompt {
    /// System instructions. Overrides the built-in default; overridden by system_file.
    pub system: Option<String>,
    /// Path to a file holding the system instructions. Read once at boot.
    pub system_file: Option<String>,
    /// Line naming the user, with {user} and {roles}.
    pub role_line: String,
    /// Line naming a user who holds no roles, with {user}.
    pub role_line_no_roles: String,
    /// Heading above recalled memories.
    pub memory_header: String,
    /// Heading above retrieved evidence.
    pub evidence_header: String,
}

impl Default for Prompt {
    fn default() -> Self {
        Self {
            system: None,
            system_file: None,
            role_line: assemble::ROLE_LINE.into(),
            role_line_no_roles: assemble::ROLE_LINE_NO_ROLES.into(),
            memory_header: assemble::MEMORY_HEADER.into(),
            evidence_header: assemble::EVIDENCE_HEADER.into(),
        }
    }
}

/// What the risk policy allows, denies, and holds for confirmation.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Policy {
    /// Roles allowed to run external_write and above. Empty denies everyone.
    pub write_roles: Vec<String>,
    /// Let tools read inside the authenticated session of the user.
    pub allow_authenticated_reads: bool,
    /// Lowest risk class that must be confirmed before it runs.
    pub confirm_from: RiskClass,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            write_roles: vec!["MANAGE_GUILD".into()],
            allow_authenticated_reads: false,
            confirm_from: RiskClass::ExternalWrite,
        }
    }
}

/// Which tools are registered.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Tools {
    /// Tool names never registered, whatever their source.
    pub disabled: Vec<String>,
    /// Register the built-in retrieval tool.
    pub knowledge_search: bool,
    /// Register the live source-query tool, when the scraper has published a registry.
    pub query_source: bool,
    /// Register the get_skill tool, when a reviewed skill exists.
    pub get_skill: bool,
}

impl Default for Tools {
    fn default() -> Self {
        Self {
            disabled: Vec::new(),
            knowledge_search: true,
            query_source: true,
            get_skill: true,
        }
    }
}

/// How a live source query runs. [tools] decides whether it is offered at all.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Query {
    /// Budget for one live query, end to end. Overrides agent.tool_timeout_secs for this tool.
    pub timeout_secs: u64,
    /// How often the engine checks whether the worker has answered.
    pub poll_ms: u64,
}

impl Default for Query {
    fn default() -> Self {
        Self {
            timeout_secs: 90,
            poll_ms: 400,
        }
    }
}

/// The chat agent. Replaces the turns that no longer fit with one turn that does.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Compaction {
    /// Compact at all. Off leaves history trimmed by dropping the oldest turns.
    pub enabled: bool,
    /// Instructions for the chat agent. Empty uses the built-in default.
    pub instructions: Option<String>,
    /// Completion budget for the compacted turn.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Wall-clock budget for the call.
    pub timeout_secs: u64,
}

impl Default for Compaction {
    fn default() -> Self {
        Self {
            enabled: true,
            instructions: None,
            max_tokens: 512,
            temperature: 0.0,
            timeout_secs: 30,
        }
    }
}

/// The gate every model response passes.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Guardrail {
    /// Check responses at all.
    pub enabled: bool,
    /// Phrases that block a response wherever they appear. Matched without case.
    pub denied_phrases: Vec<String>,
    /// Longest answer allowed. 0 removes the limit.
    pub max_answer_chars: usize,
    /// Shown in place of a blocked response.
    pub replacement: String,
}

impl Default for Guardrail {
    fn default() -> Self {
        Self {
            enabled: true,
            denied_phrases: Vec::new(),
            max_answer_chars: 8_000,
            replacement: "I cannot answer that. Ask a moderator if you need help.".into(),
        }
    }
}

/// A command in an isolated environment.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct SandboxSettings {
    /// Offer the sandbox to the model. The engine needs access to the container runtime.
    pub enabled: bool,
    /// Container runtime binary.
    pub runtime: String,
    /// Image the command runs in.
    pub image: String,
    /// Memory ceiling.
    pub memory: String,
    /// CPU ceiling.
    pub cpus: String,
    /// Process ceiling.
    pub pids: u32,
    /// Wall-clock budget for one command.
    pub timeout_secs: u64,
    /// Longest stdout or stderr handed back to the model.
    pub max_output_chars: usize,
    /// Risk class the tool declares, which is what Policy gates it by.
    pub risk: RiskClass,
    /// How long a session container stays up with nothing running in it.
    pub session_idle_secs: u64,
}

impl Default for SandboxSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            runtime: "docker".into(),
            image: "alpine:3.20".into(),
            memory: "256m".into(),
            cpus: "1".into(),
            pids: 128,
            timeout_secs: 20,
            max_output_chars: 4_000,
            risk: RiskClass::PrepareWrite,
            session_idle_secs: 900,
        }
    }
}

/// The gate on profile extraction.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Detector {
    /// First-person markers. Empty uses the built-in list.
    pub subjects: Vec<String>,
    /// Cues that mark a statement rather than a question. Empty uses the built-in list.
    pub cues: Vec<String>,
    /// Shortest turn considered.
    pub min_words: usize,
}

impl Default for Detector {
    fn default() -> Self {
        Self {
            subjects: Vec::new(),
            cues: Vec::new(),
            min_words: 4,
        }
    }
}

/// The classifier and the graph agent. Detached from the request that produced the turn.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Profile {
    /// Record what a turn states about the user.
    pub enabled: bool,
    /// What decides whether a turn is worth extracting from.
    #[serde(default)]
    pub detector: Detector,
    /// Instructions for the graph agent. Empty uses the built-in default.
    pub graph_instructions: Option<String>,
    /// Withdraw a recorded fact when a new one makes it false. Costs a model call only when a
    /// new fact collides with one already recorded.
    pub reconcile: bool,
    /// Instructions for the reconciler. Empty uses the built-in default.
    pub reconcile_instructions: Option<String>,
    /// Completion budget for the graph agent.
    pub max_tokens: u32,
    /// Wall-clock budget for classifying, extracting, and writing one turn.
    pub timeout_secs: u64,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            enabled: false,
            detector: Detector::default(),
            graph_instructions: None,
            reconcile: true,
            reconcile_instructions: None,
            max_tokens: 512,
            timeout_secs: 60,
        }
    }
}
