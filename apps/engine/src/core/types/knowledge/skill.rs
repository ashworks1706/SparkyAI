//! Skill, SkillParam, SkillStep, SkillError: saved procedures the model fetches and follows.
//!
//! A skill is parameters, an ordered list of steps, and the domain it applies to. Only reviewed
//! skills are offered, and the model follows one with the capabilities it already has.

use serde::{Deserialize, Serialize};

/// One value a skill asks the model to supply while following it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillParam {
    /// Parameter name the steps refer to.
    pub name: String,
    /// What it means, for the model.
    pub description: String,
    /// Whether the procedure cannot be followed without it.
    #[serde(default)]
    pub required: bool,
    /// An example value.
    #[serde(default)]
    pub example: Option<String>,
}

/// One step of a skill, in the order it is performed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillStep {
    /// What this step accomplishes.
    pub title: String,
    /// How to perform it, including which capability to use.
    #[serde(default)]
    pub detail: Option<String>,
}

/// A saved procedure the engine may offer, as review published it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skill {
    /// Key the model names to fetch it.
    pub key: String,
    /// Short name of the procedure.
    pub title: String,
    /// What the skill applies to.
    pub domain: String,
    /// When to use it, for the model.
    pub description: String,
    /// Values the steps need.
    pub params: Vec<SkillParam>,
    /// The steps, in order.
    pub steps: Vec<SkillStep>,
}

/// Why a skill could not be read.
#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    /// The store could not be reached, or held a row that does not parse.
    #[error("skill store: {0}")]
    Store(String),
    /// No offered skill has that key.
    #[error("no skill named {0}")]
    Unknown(String),
}
