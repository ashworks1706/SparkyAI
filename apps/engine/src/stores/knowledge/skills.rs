//! PostgreSQL adapter: SkillStore over the skills table the review workflow fills.

use async_trait::async_trait;
use serde_json::Value;
use sqlx::Row;
use sqlx::postgres::PgPool;

use crate::core::traits::knowledge::skills::SkillStore;
use crate::core::types::knowledge::skill::{Skill, SkillError, SkillParam, SkillStep};

/// Reads offered skills from PostgreSQL.
pub struct PgSkills {
    pool: PgPool,
}

/// Maps a sqlx error to a SkillError.
#[allow(clippy::needless_pass_by_value)]
fn db(e: sqlx::Error) -> SkillError {
    SkillError::Store(e.to_string())
}

impl PgSkills {
    /// Builds the store over an open pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

/// Reads one row into a Skill.
fn row_to_skill(row: &sqlx::postgres::PgRow) -> Result<Skill, SkillError> {
    let params: Value = row.try_get("params").map_err(db)?;
    let params: Vec<SkillParam> = serde_json::from_value(params)
        .map_err(|e| SkillError::Store(format!("skills.params: {e}")))?;
    let steps: Value = row.try_get("steps").map_err(db)?;
    let steps: Vec<SkillStep> = serde_json::from_value(steps)
        .map_err(|e| SkillError::Store(format!("skills.steps: {e}")))?;
    Ok(Skill {
        key: row.try_get("key").map_err(db)?,
        title: row.try_get("title").map_err(db)?,
        domain: row.try_get("domain").map_err(db)?,
        description: row.try_get("description").map_err(db)?,
        params,
        steps,
    })
}

#[async_trait]
impl SkillStore for PgSkills {
    async fn list(&self) -> Result<Vec<Skill>, SkillError> {
        let rows = sqlx::query(
            "select key, title, domain, description, params, steps
             from skills where enabled order by key",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(db)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            out.push(row_to_skill(row)?);
        }
        Ok(out)
    }

    async fn get(&self, key: &str) -> Result<Skill, SkillError> {
        let row = sqlx::query(
            "select key, title, domain, description, params, steps
             from skills where enabled and key = $1",
        )
        .bind(key)
        .fetch_optional(&self.pool)
        .await
        .map_err(db)?;
        match row {
            Some(row) => row_to_skill(&row),
            None => Err(SkillError::Unknown(key.to_owned())),
        }
    }
}
