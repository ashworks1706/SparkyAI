//! Canvas tools: read the caller's courses, assignments, and grades, in a direct message only.

pub mod client;
mod render;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{Value, json};

use crate::core::config::Canvas as CanvasConfig;
use crate::core::traits::oauth::OAuthStore;
use crate::core::traits::tools::Tool;
use crate::core::traits::tools::canvas::Canvas;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::tools::canvas::CanvasError;
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::tools::account::canvas::client::HttpCanvas;
use crate::runtime::tools::account::canvas::render::{
    announcements_output, assignment_grades_output, assignments_output, calendar_output,
    courses_output, grades_output,
};
use crate::runtime::tools::account::grant::Credentials;
use crate::runtime::tools::account::oauth::WebOAuthClient;

/// The provider key Canvas grants are stored under.
const PROVIDER: &str = "canvas";

/// Which read a Canvas tool performs.
#[derive(Debug, Clone, Copy)]
pub enum Query {
    /// Active courses.
    Courses,
    /// Assignments still to submit.
    Assignments,
    /// Current grade per course.
    Grades,
    /// Recent course announcements.
    Announcements,
    /// Upcoming calendar items.
    Calendar,
    /// Grade per graded assignment.
    AssignmentGrades,
}

/// Every read a Canvas tool is built for.
const ALL: [Query; 6] = [
    Query::Courses,
    Query::Assignments,
    Query::Grades,
    Query::Announcements,
    Query::Calendar,
    Query::AssignmentGrades,
];

impl Query {
    /// The tool name the model calls.
    fn name(self) -> &'static str {
        match self {
            Self::Courses => "canvas_courses",
            Self::Assignments => "canvas_assignments",
            Self::Grades => "canvas_grades",
            Self::Announcements => "canvas_announcements",
            Self::Calendar => "canvas_calendar",
            Self::AssignmentGrades => "canvas_assignment_grades",
        }
    }

    /// What the tool does, for the model. Every tool names the direct-message and connect gate.
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
            Self::Announcements => {
                "List recent announcements across the user's active Canvas courses. Works only in \
                 a direct message, and only after the user has connected Canvas."
            }
            Self::Calendar => {
                "List the user's upcoming Canvas calendar events, soonest first. Works only in a \
                 direct message, and only after the user has connected Canvas."
            }
            Self::AssignmentGrades => {
                "List the user's score on each graded Canvas assignment, by course. Works only in \
                 a direct message, and only after the user has connected Canvas."
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
        let Some(token) = self.creds.resolve(ctx).await? else {
            return Err(ToolError::Failed(
                "You have not connected Canvas. Send /login in a direct message to connect it."
                    .to_owned(),
            ));
        };
        match self.query {
            Query::Courses => {
                let courses = self.client.courses(&token).await.map_err(failed)?;
                Ok(courses_output(courses, self.max_items))
            }
            Query::Assignments => {
                let assignments = self.client.assignments(&token).await.map_err(failed)?;
                Ok(assignments_output(assignments, self.max_items))
            }
            Query::Grades => {
                let grades = self.client.grades(&token).await.map_err(failed)?;
                Ok(grades_output(grades, self.max_items))
            }
            Query::Announcements => {
                let items = self.client.announcements(&token).await.map_err(failed)?;
                Ok(announcements_output(items, self.max_items))
            }
            Query::Calendar => {
                let items = self.client.calendar(&token).await.map_err(failed)?;
                Ok(calendar_output(items, self.max_items))
            }
            Query::AssignmentGrades => {
                let items = self
                    .client
                    .assignment_grades(&token)
                    .await
                    .map_err(failed)?;
                Ok(assignment_grades_output(items, self.max_items))
            }
        }
    }
}

/// The Canvas tools, over the API client, the per-user grant store, and the client used to refresh.
pub fn tools(
    cfg: &CanvasConfig,
    store: Arc<dyn OAuthStore>,
    oauth: Option<Arc<WebOAuthClient>>,
) -> Result<Vec<Arc<dyn Tool>>, CanvasError> {
    let client: Arc<dyn Canvas> = Arc::new(HttpCanvas::new(
        &cfg.base_url,
        Duration::from_secs(cfg.timeout_secs),
        cfg.max_items,
    )?);
    let fallback = SecretString::from(cfg.access_token.expose_secret().to_owned());
    let creds = Arc::new(Credentials::new(store, oauth, PROVIDER, fallback));
    Ok(ALL
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
