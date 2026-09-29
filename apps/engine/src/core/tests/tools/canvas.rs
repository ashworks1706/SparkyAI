//! Canvas tools: direct-message gating, connection gating, error mapping, and the replies.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use secrecy::SecretString;
use serde_json::json;

use crate::core::traits::tools::Tool;
use crate::core::traits::tools::canvas::Canvas;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::tools::ToolError;
use crate::core::types::tools::canvas::{Assignment, CanvasError, Course, CourseGrade};
use crate::runtime::tools::canvas::{CanvasTool, Credentials, Query};

/// A Canvas double that answers with canned rows, or a set error status.
struct FakeCanvas {
    courses: Vec<Course>,
    assignments: Vec<Assignment>,
    grades: Vec<CourseGrade>,
    error: Option<u16>,
}

impl FakeCanvas {
    fn empty() -> Self {
        Self {
            courses: Vec::new(),
            assignments: Vec::new(),
            grades: Vec::new(),
            error: None,
        }
    }
}

#[async_trait]
impl Canvas for FakeCanvas {
    async fn courses(&self, _token: &SecretString) -> Result<Vec<Course>, CanvasError> {
        match self.error {
            Some(status) => Err(CanvasError::Refused(status)),
            None => Ok(self.courses.clone()),
        }
    }
    async fn assignments(&self, _token: &SecretString) -> Result<Vec<Assignment>, CanvasError> {
        match self.error {
            Some(status) => Err(CanvasError::Refused(status)),
            None => Ok(self.assignments.clone()),
        }
    }
    async fn grades(&self, _token: &SecretString) -> Result<Vec<CourseGrade>, CanvasError> {
        match self.error {
            Some(status) => Err(CanvasError::Refused(status)),
            None => Ok(self.grades.clone()),
        }
    }
}

fn dm() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5)).with_visibility(Visibility::Private)
}

fn server() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5)).with_visibility(Visibility::Public)
}

fn tool(query: Query, fake: FakeCanvas, token: &str) -> CanvasTool {
    CanvasTool::new(
        query,
        Arc::new(fake),
        Arc::new(Credentials::new(SecretString::from(token.to_owned()))),
        20,
    )
}

#[tokio::test]
async fn canvas_is_refused_outside_a_direct_message() {
    let out = tool(Query::Courses, FakeCanvas::empty(), "tok")
        .call(&server(), json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("direct message")),
        "{out:?}"
    );
}

#[tokio::test]
async fn an_unconnected_user_is_told_to_connect() {
    let out = tool(Query::Courses, FakeCanvas::empty(), "")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("not connected")),
        "{out:?}"
    );
}

#[tokio::test]
async fn courses_are_listed_with_name_and_code() {
    let fake = FakeCanvas {
        courses: vec![Course {
            id: 1,
            name: "Intro to AI".into(),
            code: "CSE 471".into(),
        }],
        ..FakeCanvas::empty()
    };
    let out = tool(Query::Courses, fake, "tok")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Ok(o) if o.content.contains("Intro to AI") && o.content.contains("CSE 471")),
        "{out:?}"
    );
}

#[tokio::test]
async fn assignments_carry_course_due_date_and_points() {
    let due = DateTime::from_timestamp(1_760_000_000, 0).unwrap_or_else(Utc::now);
    let fake = FakeCanvas {
        assignments: vec![Assignment {
            name: "Project 1".into(),
            course: "CSE 471".into(),
            due_at: Some(due),
            points: Some(100.0),
            url: Some("https://canvas.asu.edu/1".into()),
        }],
        ..FakeCanvas::empty()
    };
    let out = tool(Query::Assignments, fake, "tok")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Ok(o) if o.content.contains("Project 1")
            && o.content.contains("CSE 471") && o.content.contains("pts")),
        "{out:?}"
    );
}

#[tokio::test]
async fn grades_show_the_letter_and_score() {
    let fake = FakeCanvas {
        grades: vec![CourseGrade {
            course: "CSE 471".into(),
            score: Some(92.5),
            grade: Some("A".into()),
        }],
        ..FakeCanvas::empty()
    };
    let out = tool(Query::Grades, fake, "tok")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Ok(o) if o.content.contains("CSE 471")
            && o.content.contains('A') && o.content.contains("92.5")),
        "{out:?}"
    );
}

#[tokio::test]
async fn an_expired_token_is_named_as_such() {
    let fake = FakeCanvas {
        error: Some(401),
        ..FakeCanvas::empty()
    };
    let out = tool(Query::Courses, fake, "tok")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("expired")),
        "{out:?}"
    );
}
