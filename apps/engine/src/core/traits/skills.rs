//! SkillStore trait: the saved procedures review has offered.

use async_trait::async_trait;

use crate::core::types::skill::{Skill, SkillError};

/// Reads the skills the model may follow.
///
/// Only reviewed skills are offered: nothing here writes, and a skill is never promoted from a
/// trace without a person in the loop.
#[async_trait]
pub trait SkillStore: Send + Sync {
    /// Skills currently offered, by key. Read once at boot to build the tool.
    async fn list(&self) -> Result<Vec<Skill>, SkillError>;

    /// One offered skill, or Unknown if no offered skill has that key.
    async fn get(&self, key: &str) -> Result<Skill, SkillError>;
}
