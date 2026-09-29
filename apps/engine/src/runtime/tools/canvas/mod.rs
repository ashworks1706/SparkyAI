//! Canvas tools: read the caller's courses, assignments, and grades, in a direct message only.

pub mod client;

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{Value, json};

use crate::core::config::Canvas as CanvasConfig;
use crate::core::traits::tools::Tool;
use crate::core::traits::tools::canvas::Canvas;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::tools::canvas::{Assignment, CanvasError, Course, CourseGrade};
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::tools::canvas::client::HttpCanvas;
use crate::runtime::tools::structured;

/// The Canvas token used for a request.
///
/// Today it is the shared token from configuration. The per-user grant store of roadmap phase 8
/// resolves the caller's own token here, keyed by ctx.user_id, before falling back to this one.
pub struct Credentials {
    token: SecretString,
}

impl Credentials {
    /// Holds the configured token.
    pub fn new(token: SecretString) -> Self {
        Self { token }
    }

    /// The token to use for the caller, or None when Canvas is not connected.
    pub fn resolve(&self, _ctx: &RequestContext) -> Option<&SecretString> {
        if self.token.expose_secret().trim().is_empty() {
            None
        } else {
            Some(&self.token)
        }
    }
}

/// Which read a Canvas tool performs.
#[derive(Debug, Clone, Copy)]
pub enum Query {
    /// Active courses.
    Courses,
    /// Assignments still to submit.
    Assignments,
    /// Current grade per course.
    Grades,
}

impl Query {
    /// The tool name the model calls.
    fn name(self) -> &'static str {
        match self {
            Self::Courses => "canvas_courses",
            Self::Assignments => "canvas_assignments",
            Self::Grades => "canvas_grades",
        }
    }

    /// What the tool does, for the model.
    fn description(self) -> &'static str {
        match self {
            Self::Courses => {
                "List the ASU Canvas courses the user is enrolled in this term. Works only in a \
                 direct message, and only after the user has connected Canvas."
            }
            Self::Assignments => {
                "List the user's upcoming Canvas assignments with their due dates, soonest first. \
                 Works only in a direct message, and only after the user has connected Canvas."
            }
            Self::Grades => {
                "List the user's current grade in each active Canvas course. Works only in a \
                 direct message, and only after the user has connected Canvas."
            }
        }
    }
}

/// One Canvas read, offered to the model.
pub struct CanvasTool {
    query: Query,
    client: Arc<dyn Canvas>,
    creds: Arc<Credentials>,
    max_items: usize,
}

impl CanvasTool {
    /// Builds a tool over the client and the shared credentials.
    pub fn new(
        query: Query,
        client: Arc<dyn Canvas>,
        creds: Arc<Credentials>,
        max_items: usize,
    ) -> Self {
        Self {
            query,
            client,
            creds,
            max_items,
        }
    }
}

/// Maps a Canvas failure to the text the model reads.
fn failed(error: CanvasError) -> ToolError {
    match error {
        CanvasError::Refused(401 | 403) => {
            ToolError::Failed("Canvas rejected the token; it may have expired.".to_owned())
        }
        other => ToolError::Failed(other.to_string()),
    }
}

#[async_trait]
impl Tool for CanvasTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.query.name().to_owned(),
            description: self.query.description().to_owned(),
            parameters: json!({ "type": "object", "properties": {} }),
            risk: RiskClass::ReadAuthenticated,
            sequential: false,
            timeout_secs: None,
        }
    }

    async fn call(&self, ctx: &RequestContext, _args: Value) -> Result<ToolOutput, ToolError> {
        if ctx.visibility != Visibility::Private {
            return Err(ToolError::Failed(
                "Canvas is available only in a direct message with me, not in a server.".to_owned(),
            ));
        }
        let token = self.creds.resolve(ctx).ok_or_else(|| {
            ToolError::Failed(
                "Canvas is not connected. Ask the operator to set canvas.access_token.".to_owned(),
            )
        })?;
        match self.query {
            Query::Courses => {
                let courses = self.client.courses(token).await.map_err(failed)?;
                Ok(courses_output(courses, self.max_items))
            }
            Query::Assignments => {
                let assignments = self.client.assignments(token).await.map_err(failed)?;
                Ok(assignments_output(assignments, self.max_items))
            }
            Query::Grades => {
                let grades = self.client.grades(token).await.map_err(failed)?;
                Ok(grades_output(grades, self.max_items))
            }
        }
    }
}

/// A due date as the model reads it, or a note that none is set.
fn due(assignment: &Assignment) -> String {
    match assignment.due_at {
        Some(at) => at.format("%b %d, %Y %H:%M UTC").to_string(),
        None => "no due date".to_owned(),
    }
}

/// The courses reply.
fn courses_output(courses: Vec<Course>, max: usize) -> ToolOutput {
    let shown: Vec<Course> = courses.into_iter().take(max).collect();
    let mut text = if shown.is_empty() {
        "No active Canvas courses.".to_owned()
    } else {
        let mut lines = format!("{} active courses:", shown.len());
        for c in &shown {
            let _ = write!(lines, "\n- {} ({})", c.name, c.code);
        }
        lines
    };
    text.push('\n');
    ToolOutput {
        content: text.trim_end().to_owned(),
        data: structured(&shown),
        sources: Vec::new(),
    }
}

/// The assignments reply.
fn assignments_output(assignments: Vec<Assignment>, max: usize) -> ToolOutput {
    let shown: Vec<Assignment> = assignments.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No upcoming assignments to submit.".to_owned()
    } else {
        let mut lines = format!("{} upcoming assignments:", shown.len());
        for a in &shown {
            let points = a.points.map(|p| format!(", {p} pts")).unwrap_or_default();
            let link = a
                .url
                .as_deref()
                .map(|u| format!(" {u}"))
                .unwrap_or_default();
            let _ = write!(
                lines,
                "\n- {} ({}) due {}{}{}",
                a.name,
                a.course,
                due(a),
                points,
                link
            );
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}

/// The grades reply.
fn grades_output(grades: Vec<CourseGrade>, max: usize) -> ToolOutput {
    let shown: Vec<CourseGrade> = grades.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No grades available.".to_owned()
    } else {
        let mut lines = "Current grades:".to_owned();
        for g in &shown {
            let standing = match (&g.grade, g.score) {
                (Some(letter), Some(score)) => format!("{letter} ({score})"),
                (Some(letter), None) => letter.clone(),
                (None, Some(score)) => score.to_string(),
                (None, None) => "not graded yet".to_owned(),
            };
            let _ = write!(lines, "\n- {}: {}", g.course, standing);
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}

/// The Canvas tools, built over one shared client and the configured token.
pub fn tools(cfg: &CanvasConfig) -> Result<Vec<Arc<dyn Tool>>, CanvasError> {
    let client: Arc<dyn Canvas> = Arc::new(HttpCanvas::new(
        &cfg.base_url,
        Duration::from_secs(cfg.timeout_secs),
        cfg.max_items,
    )?);
    let creds = Arc::new(Credentials::new(SecretString::from(
        cfg.access_token.expose_secret().to_owned(),
    )));
    Ok([Query::Courses, Query::Assignments, Query::Grades]
        .into_iter()
        .map(|q| {
            Arc::new(CanvasTool::new(
                q,
                client.clone(),
                creds.clone(),
                cfg.max_items,
            )) as Arc<dyn Tool>
        })
        .collect())
}
