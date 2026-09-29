//! Canvas tools: DM gating, per-user grant resolution, error mapping, and the replies.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use secrecy::SecretString;
use serde_json::json;

use crate::core::traits::oauth::OAuthStore;
use crate::core::traits::tools::Tool;
use crate::core::traits::tools::canvas::Canvas;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::store::StoreError;
use crate::core::types::tools::ToolError;
use crate::core::types::tools::canvas::{
    Announcement, Assignment, AssignmentGrade, CalendarEvent, CanvasError, Course, CourseGrade,
};
use crate::core::types::tools::oauth::{Consent, OAuthTokens};
use crate::runtime::tools::canvas::{CanvasTool, Query};
use crate::runtime::tools::grant::Credentials;

/// A Canvas double that answers with canned rows, or a set error status.
#[derive(Default)]
struct FakeCanvas {
    courses: Vec<Course>,
    assignments: Vec<Assignment>,
    grades: Vec<CourseGrade>,
    announcements: Vec<Announcement>,
    calendar: Vec<CalendarEvent>,
    assignment_grades: Vec<AssignmentGrade>,
    error: Option<u16>,
}

impl FakeCanvas {
    fn empty() -> Self {
        Self::default()
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
    async fn announcements(&self, _token: &SecretString) -> Result<Vec<Announcement>, CanvasError> {
        match self.error {
            Some(status) => Err(CanvasError::Refused(status)),
            None => Ok(self.announcements.clone()),
        }
    }
    async fn calendar(&self, _token: &SecretString) -> Result<Vec<CalendarEvent>, CanvasError> {
        match self.error {
            Some(status) => Err(CanvasError::Refused(status)),
            None => Ok(self.calendar.clone()),
        }
    }
    async fn assignment_grades(
        &self,
        _token: &SecretString,
    ) -> Result<Vec<AssignmentGrade>, CanvasError> {
        match self.error {
            Some(status) => Err(CanvasError::Refused(status)),
            None => Ok(self.assignment_grades.clone()),
        }
    }
}

/// A grant store double holding one caller's grant.
struct FakeGrants {
    grant: Option<OAuthTokens>,
}

#[async_trait]
impl OAuthStore for FakeGrants {
    async fn save_grant(
        &self,
        _tenant: &str,
        _user: &str,
        _provider: &str,
        _tokens: &OAuthTokens,
    ) -> Result<(), StoreError> {
        Ok(())
    }
    async fn load_grant(
        &self,
        _tenant: &str,
        _user: &str,
        _provider: &str,
    ) -> Result<Option<OAuthTokens>, StoreError> {
        Ok(self.grant.clone())
    }
    async fn delete_grant(
        &self,
        _tenant: &str,
        _user: &str,
        _provider: &str,
    ) -> Result<bool, StoreError> {
        Ok(false)
    }
    async fn begin_consent(
        &self,
        _state: &str,
        _tenant: &str,
        _user: &str,
        _provider: &str,
        _ttl: Duration,
    ) -> Result<(), StoreError> {
        Ok(())
    }
    async fn take_consent(&self, _state: &str) -> Result<Option<Consent>, StoreError> {
        Ok(None)
    }
}

fn grant(token: &str) -> OAuthTokens {
    OAuthTokens {
        access_token: SecretString::from(token.to_owned()),
        refresh_token: None,
        scopes: Vec::new(),
        expires_at: None,
    }
}

fn dm() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5)).with_visibility(Visibility::Private)
}

fn server() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5)).with_visibility(Visibility::Public)
}

/// A tool whose credentials come from a stored grant, else the shared fallback token.
fn tool(query: Query, fake: FakeCanvas, stored: Option<OAuthTokens>, fallback: &str) -> CanvasTool {
    let creds = Credentials::new(
        Arc::new(FakeGrants { grant: stored }),
        None,
        "canvas",
        SecretString::from(fallback.to_owned()),
    );
    CanvasTool::new(query, Arc::new(fake), Arc::new(creds), 20)
}

#[tokio::test]
async fn canvas_is_refused_outside_a_direct_message() {
    let out = tool(Query::Courses, FakeCanvas::empty(), None, "tok")
        .call(&server(), json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("direct message")),
        "{out:?}"
    );
}

#[tokio::test]
async fn an_unconnected_user_is_told_to_log_in() {
    let out = tool(Query::Courses, FakeCanvas::empty(), None, "")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("/login")),
        "{out:?}"
    );
}

#[tokio::test]
async fn a_stored_grant_is_used_when_present() {
    let fake = FakeCanvas {
        courses: vec![Course {
            id: 1,
            name: "Intro to AI".into(),
            code: "CSE 471".into(),
        }],
        ..FakeCanvas::empty()
    };
    let out = tool(Query::Courses, fake, Some(grant("tok")), "")
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
    let out = tool(Query::Assignments, fake, None, "tok")
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
    let out = tool(Query::Grades, fake, None, "tok")
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
    let out = tool(Query::Courses, fake, None, "tok")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Err(ToolError::Failed(m)) if m.contains("expired")),
        "{out:?}"
    );
}

#[tokio::test]
async fn announcements_carry_title_and_course() {
    let fake = FakeCanvas {
        announcements: vec![Announcement {
            title: "Midterm moved".into(),
            course: "CSE 471".into(),
            posted_at: None,
            body: Some("The midterm is now next week.".into()),
            url: Some("https://canvas.asu.edu/a/1".into()),
        }],
        ..FakeCanvas::empty()
    };
    let out = tool(Query::Announcements, fake, None, "tok")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Ok(o) if o.content.contains("Midterm moved") && o.content.contains("CSE 471")),
        "{out:?}"
    );
}

#[tokio::test]
async fn calendar_lists_upcoming_events_with_location() {
    let fake = FakeCanvas {
        calendar: vec![CalendarEvent {
            title: "Lecture".into(),
            start_at: None,
            location: Some("COOR 170".into()),
            url: None,
        }],
        ..FakeCanvas::empty()
    };
    let out = tool(Query::Calendar, fake, None, "tok")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Ok(o) if o.content.contains("Lecture") && o.content.contains("COOR 170")),
        "{out:?}"
    );
}

#[tokio::test]
async fn assignment_grades_show_score_over_points() {
    let fake = FakeCanvas {
        assignment_grades: vec![AssignmentGrade {
            course: "CSE 471".into(),
            name: "HW1".into(),
            score: Some(18.0),
            points: Some(20.0),
            grade: None,
        }],
        ..FakeCanvas::empty()
    };
    let out = tool(Query::AssignmentGrades, fake, None, "tok")
        .call(&dm(), json!({}))
        .await;
    assert!(
        matches!(&out, Ok(o) if o.content.contains("HW1")
            && o.content.contains("18") && o.content.contains("20")),
        "{out:?}"
    );
}
