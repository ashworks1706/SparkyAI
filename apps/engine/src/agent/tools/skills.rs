//! ReadPublic: fetch one saved procedure so the model can follow it.
//!
//! The tool returns steps, not results. Whatever the steps call for is done with the
//! capabilities the model already has, under their own risk classes.

use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::core::traits::skills::SkillStore;
use crate::core::traits::tool::Tool;
use crate::core::types::context::RequestContext;
use crate::core::types::skill::{Skill, SkillError};
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};

/// Fetches one skill per call.
pub struct GetSkillTool {
    skills: Arc<dyn SkillStore>,
    keys: Vec<String>,
    definition: ToolDefinition,
}

/// The description the model reads: which procedures exist, by key and domain.
pub fn describe(skills: &[Skill]) -> String {
    let mut text = String::from(
        "Fetch a saved procedure and follow its steps with the tools you already have. Use it \
         when the request matches one of these; the steps come back as text, nothing runs on \
         your behalf. Skills:",
    );
    if skills.is_empty() {
        text.push_str("\n\nnone offered.");
        return text;
    }
    for skill in skills {
        let _ = write!(
            text,
            "\n\n`{}` ({}) — {}: {}",
            skill.key,
            skill.domain.trim(),
            skill.title.trim(),
            skill.description.trim()
        );
    }
    text
}

/// Renders a skill as the ordered text the model follows.
pub fn render(skill: &Skill) -> String {
    let mut text = format!(
        "Skill `{}` — {} ({})\n{}",
        skill.key,
        skill.title.trim(),
        skill.domain.trim(),
        skill.description.trim()
    );
    if !skill.params.is_empty() {
        text.push_str("\n\nNeeds:");
        for p in &skill.params {
            let _ = write!(
                text,
                "\n  {}{} — {}",
                p.name,
                if p.required { " (required)" } else { "" },
                p.description.trim()
            );
            if let Some(example) = &p.example {
                let _ = write!(text, " e.g. {example}");
            }
        }
    }
    text.push_str("\n\nSteps, in order:");
    for (i, step) in skill.steps.iter().enumerate() {
        let _ = write!(text, "\n  {}. {}", i + 1, step.title.trim());
        if let Some(detail) = &step.detail {
            let _ = write!(text, " — {}", detail.trim());
        }
    }
    text
}

impl GetSkillTool {
    /// Builds the tool over the skills review currently offers.
    pub fn new(skills: Arc<dyn SkillStore>, offered: &[Skill]) -> Self {
        let keys: Vec<String> = offered.iter().map(|s| s.key.clone()).collect();
        Self {
            skills,
            keys: keys.clone(),
            definition: ToolDefinition {
                name: "get_skill".into(),
                description: describe(offered),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "enum": keys,
                            "description": "Which skill to fetch."
                        }
                    },
                    "required": ["key"]
                }),
                risk: RiskClass::ReadPublic,
                sequential: false,
                timeout_secs: None,
            },
        }
    }

    /// The refusal an unknown key gets: the keys that do exist, so the model can correct itself.
    fn unknown(&self, key: &str) -> ToolError {
        if self.keys.is_empty() {
            return ToolError::InvalidArguments(format!("no skill named {key}; none are offered"));
        }
        ToolError::InvalidArguments(format!(
            "no skill named {key}; the skills are: {}",
            self.keys.join(", ")
        ))
    }
}

#[async_trait]
impl Tool for GetSkillTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn call(&self, _ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        let key = args
            .get("key")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .ok_or_else(|| ToolError::InvalidArguments("key is required".into()))?;
        let skill = self.skills.get(key).await.map_err(|e| match e {
            SkillError::Unknown(key) => self.unknown(&key),
            store @ SkillError::Store(_) => ToolError::Failed(store.to_string()),
        })?;
        Ok(ToolOutput {
            content: render(&skill),
            data: serde_json::to_value(&skill).ok(),
        })
    }
}
