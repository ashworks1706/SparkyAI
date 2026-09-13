//! SkillStore trait: the saved procedures review has offered.

use async_trait::async_trait;

use crate::core::types::knowledge::skill::{Skill, SkillError};

/// Reads the reviewed skills the model may follow. Read-only.
#[async_trait]
pub trait SkillStore: Send + Sync {
    /// Skills currently offered, by key. Read once at boot to build the tool.
    async fn list(&self) -> Result<Vec<Skill>, SkillError>;

    /// One offered skill, or Unknown if no offered skill has that key.
    async fn get(&self, key: &str) -> Result<Skill, SkillError>;
}
