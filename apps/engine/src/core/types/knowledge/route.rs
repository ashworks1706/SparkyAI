//! Route: whether a question is worth retrieving evidence for, and why it was not.

use serde::{Deserialize, Serialize};

/// Why retrieval was skipped for a question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Skipped {
    /// Small talk: the question asks for no ASU fact the index could hold.
    Chitchat,
    /// The answer has to be current, so a live source tool answers it.
    Live,
}

/// What the router decided about retrieval for one question.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// Retrieve evidence before the first model call.
    #[default]
    Retrieve,
    /// Skip retrieval for this reason.
    Skip(Skipped),
}

impl Route {
    /// The reason retrieval was skipped, or None when it ran.
    pub fn skipped(self) -> Option<Skipped> {
        match self {
            Self::Retrieve => None,
            Self::Skip(reason) => Some(reason),
        }
    }
}
