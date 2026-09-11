//! The get_skill tool: what the model is shown, what a fetched skill reads like, and what an
//! unknown or unreviewed key gets back.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use crate::agent::tools::knowledge::skills::{GetSkillTool, describe};
use crate::core::tests::support::ctx;
use crate::core::traits::knowledge::skills::SkillStore;
use crate::core::traits::tools::Tool;
use crate::core::types::knowledge::skill::{Skill, SkillError, SkillParam, SkillStep};
use crate::core::types::tools::{RiskClass, ToolError};

/// A skill store holding reviewed and unreviewed skills, offering only the reviewed ones.
struct FakeSkills {
    rows: Vec<(Skill, bool)>,
}

impl FakeSkills {
    fn new(rows: Vec<(Skill, bool)>) -> Self {
        Self { rows }
    }

    fn enabled(&self) -> Vec<Skill> {
        self.rows
            .iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(skill, _)| skill.clone())
            .collect()
    }
}

#[async_trait]
impl SkillStore for FakeSkills {
    async fn list(&self) -> Result<Vec<Skill>, SkillError> {
        Ok(self.enabled())
    }

    async fn get(&self, key: &str) -> Result<Skill, SkillError> {
        self.enabled()
            .into_iter()
            .find(|skill| skill.key == key)
            .ok_or_else(|| SkillError::Unknown(key.to_owned()))
    }
}

fn add_drop() -> Skill {
    Skill {
        key: "add_drop_a_class".into(),
        title: "Add or drop a class".into(),
        domain: "registrar".into(),
        description: "Change enrollment inside the deadline.".into(),
        params: vec![
            SkillParam {
                name: "term".into(),
                description: "Term being changed.".into(),
                required: true,
                example: Some("Fall 2026".into()),
            },
            SkillParam {
                name: "reason".into(),
                description: "Why, if the student gave one.".into(),
                required: false,
                example: None,
            },
        ],
        steps: vec![
            SkillStep {
                title: "Confirm the deadline for the term".into(),
                detail: Some("query_source academic_calendar".into()),
            },
            SkillStep {
                title: "Check the enrollment holds".into(),
                detail: None,
            },
            SkillStep {
                title: "Hand the student the registrar link".into(),
                detail: None,
            },
        ],
    }
}

fn draft_appeal() -> Skill {
    Skill {
        key: "draft_grade_appeal".into(),
        title: "Draft a grade appeal".into(),
        domain: "advising".into(),
        description: "Prepare the letter the student files.".into(),
        params: Vec::new(),
        steps: vec![SkillStep {
            title: "Collect the course and the grade".into(),
            detail: None,
        }],
    }
}

fn tool(store: FakeSkills) -> GetSkillTool {
    let offered = store.enabled();
    GetSkillTool::new(Arc::new(store), &offered)
}

#[test]
fn the_description_names_every_offered_skill_by_key_and_domain() {
    let text = describe(&[add_drop(), draft_appeal()]);
    assert!(text.contains("`add_drop_a_class`"), "{text}");
    assert!(text.contains("registrar"), "{text}");
    assert!(text.contains("`draft_grade_appeal`"), "{text}");
    assert!(text.contains("advising"), "{text}");

    // A skill is picked by key, so adding one must not add a schema.
    let definition = tool(FakeSkills::new(vec![(add_drop(), true)])).definition();
    let props = &definition.parameters["properties"];
    assert_eq!(props.as_object().map(serde_json::Map::len), Some(1));
    assert_eq!(props["key"]["enum"], json!(["add_drop_a_class"]));
    assert_eq!(definition.risk, RiskClass::ReadPublic);
}

#[tokio::test]
async fn a_known_key_comes_back_as_steps_in_order() {
    let out = tool(FakeSkills::new(vec![(add_drop(), true)]))
        .call(&ctx(), json!({"key": "add_drop_a_class"}))
        .await;
    let Ok(output) = out else {
        unreachable!("the skill is offered")
    };
    let text = output.content;
    // Order is the procedure: a skill read out of order is a different procedure.
    let Some(first) = text.find("1. Confirm the deadline") else {
        unreachable!("step one is numbered: {text}")
    };
    let Some(second) = text.find("2. Check the enrollment holds") else {
        unreachable!("step two is numbered: {text}")
    };
    let Some(third) = text.find("3. Hand the student") else {
        unreachable!("step three is numbered: {text}")
    };
    assert!(first < second && second < third, "{text}");
    // The parameters the steps need travel with them.
    assert!(text.contains("term (required)"), "{text}");
    assert!(text.contains("e.g. Fall 2026"), "{text}");
    assert!(!text.contains("reason (required)"), "{text}");
}

#[tokio::test]
async fn an_unknown_key_is_refused_with_the_keys_that_exist() {
    let store = FakeSkills::new(vec![(add_drop(), true), (draft_appeal(), true)]);
    let err = tool(store).call(&ctx(), json!({"key": "made_up"})).await;
    // InvalidArguments is fed back for the model to correct, so it must carry the real keys.
    match err {
        Err(ToolError::InvalidArguments(reason)) => {
            assert!(reason.contains("add_drop_a_class"), "{reason}");
            assert!(reason.contains("draft_grade_appeal"), "{reason}");
        }
        other => unreachable!("expected a correctable refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn an_unreviewed_skill_is_neither_listed_nor_fetchable() {
    let store = FakeSkills::new(vec![(add_drop(), true), (draft_appeal(), false)]);
    let Ok(offered) = store.list().await else {
        unreachable!("the store answered")
    };
    assert_eq!(offered.len(), 1);
    assert_eq!(offered[0].key, "add_drop_a_class");

    let text = describe(&offered);
    assert!(!text.contains("draft_grade_appeal"), "{text}");

    // Naming it directly does not reach it either.
    let err = tool(store)
        .call(&ctx(), json!({"key": "draft_grade_appeal"}))
        .await;
    assert!(
        matches!(err, Err(ToolError::InvalidArguments(_))),
        "{err:?}"
    );
}
