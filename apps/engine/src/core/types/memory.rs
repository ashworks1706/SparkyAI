//! `MemoryKind`, `Memory`, `MemoryCandidate`, `MemoryQuery`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// What kind of memory this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    /// A useful event from a prior interaction.
    Episodic,
    /// A stable inferred fact.
    Semantic,
    /// Approved preferences, interests, goals.
    Profile,
    /// State to continue a multi-step job.
    Task,
}

impl MemoryKind {
    /// Database column value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Profile => "profile",
            Self::Task => "task",
        }
    }

    /// Parses a database column value.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "episodic" => Some(Self::Episodic),
            "semantic" => Some(Self::Semantic),
            "profile" => Some(Self::Profile),
            "task" => Some(Self::Task),
            _ => None,
        }
    }
}

/// A stored memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Row id.
    pub id: Uuid,
    /// Kind.
    pub kind: MemoryKind,
    /// The remembered text.
    pub content: String,
    /// Confidence at write time, 0–1.
    pub confidence: f32,
    /// When it was written.
    pub created_at: DateTime<Utc>,
    /// When it stops being recalled.
    pub expires_at: Option<DateTime<Utc>>,
}

/// A memory the agent wants to write. The store applies the write policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCandidate {
    /// Kind.
    pub kind: MemoryKind,
    /// The text.
    pub content: String,
    /// Confidence, 0–1.
    pub confidence: f32,
    /// Required expiry.
    pub expires_at: DateTime<Utc>,
}

/// Recall parameters.
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    /// Restrict to these kinds; empty means all.
    pub kinds: Vec<MemoryKind>,
    /// Maximum rows.
    pub limit: usize,
}
