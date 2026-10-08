//! What the engine remembers about a user, and forgetting it.

use serde::{Deserialize, Serialize};

/// Body of POST /profile/list.
#[derive(Debug, Serialize)]
pub struct ProfileRequest {
    /// Discord user id.
    pub user_id: String,
    /// Guild id.
    pub tenant_id: String,
}

/// What the engine remembers about one user.
#[derive(Debug, Default, Deserialize)]
pub struct ProfileList {
    /// Things remembered.
    #[serde(default)]
    pub nodes: Vec<ProfileNode>,
    /// How those things relate.
    #[serde(default)]
    pub relations: Vec<ProfileRelation>,
}

/// One remembered thing.
#[derive(Debug, Clone, Deserialize)]
pub struct ProfileNode {
    /// Category, such as course or club.
    pub kind: String,
    /// Name of the thing.
    pub label: String,
    /// Belief from 0 to 1.
    pub confidence: f64,
}

/// One remembered relation between two things.
#[derive(Debug, Clone, Deserialize)]
pub struct ProfileRelation {
    /// Label of the first thing.
    pub subject: String,
    /// How they relate.
    pub relation: String,
    /// Label of the second thing.
    pub object: String,
    /// Belief from 0 to 1.
    pub confidence: f64,
}

/// Body of POST /profile/forget. No label removes everything.
#[derive(Debug, Serialize)]
pub struct ForgetRequest {
    /// Discord user id.
    pub user_id: String,
    /// Guild id.
    pub tenant_id: String,
    /// The one thing to forget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Reply to POST /profile/forget.
#[derive(Debug, Deserialize)]
pub struct ForgetResponse {
    /// Items removed.
    pub removed: u64,
}
