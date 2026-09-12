//! How a turn becomes a profile.

use serde::Deserialize;

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
    /// Lowest extraction confidence written to the graph, 0 to 1.
    pub min_confidence: f32,
    /// Nodes and relations POST /profile/list returns, each.
    pub list_limit: usize,
    /// Wall-clock budget for one /profile/list or /profile/forget call.
    pub request_timeout_secs: u64,
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
            min_confidence: 0.5,
            list_limit: 50,
            request_timeout_secs: 30,
        }
    }
}
