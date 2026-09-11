//! ProfileEntity, ProfileNode, ProfileFact, ProfileError: the user profile graph.
//!
//! A fact is what extraction produces from one turn. Storing it writes a node per entity and an
//! edge for the relation between them, so recall can start from a relation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Full confidence, the value an extraction that says nothing about it carries.
fn certain() -> f32 {
    1.0
}

/// One entity named in a fact, before it is matched to a stored node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileEntity {
    /// What sort of thing it is: person, course, club, place, topic.
    pub kind: String,
    /// How it is named, as the text the node is embedded from.
    pub label: String,
}

/// One stored entity in a user's graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileNode {
    /// Row id.
    pub id: Uuid,
    /// What sort of thing it is.
    pub kind: String,
    /// How it is named.
    pub label: String,
    /// Confidence at write time, 0 to 1.
    pub confidence: f32,
    /// When it first entered the graph.
    pub created_at: DateTime<Utc>,
    /// When extraction last confirmed it.
    pub updated_at: DateTime<Utc>,
}

/// What extraction produces from one turn, before it is stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileFact {
    /// Entity the relation starts at.
    pub subject: ProfileEntity,
    /// What the relation is.
    pub relation: String,
    /// Entity the relation points to.
    pub object: ProfileEntity,
    /// How sure extraction is, 0 to 1.
    #[serde(default = "certain")]
    pub confidence: f32,
}

/// Why a profile graph read, write, or extraction did not happen.
#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    /// The database rejected or could not run the operation.
    #[error("profile graph store: {0}")]
    Store(String),
    /// A node label could not be embedded.
    #[error("profile graph embedding: {0}")]
    Embedding(String),
    /// The model answered with something the extraction shape does not fit.
    #[error("profile extraction malformed: {0}")]
    Malformed(String),
    /// The model call itself failed.
    #[error(transparent)]
    Model(#[from] crate::core::types::model::ModelError),
}

impl From<&ProfileNode> for crate::core::types::memory::Memory {
    /// A recalled node reaches the prompt through the memory section.
    fn from(node: &ProfileNode) -> Self {
        Self {
            id: node.id,
            kind: crate::core::types::memory::MemoryKind::Profile,
            content: format!("{}: {}", node.kind, node.label),
            confidence: node.confidence,
            created_at: node.created_at,
            expires_at: None,
        }
    }
}
