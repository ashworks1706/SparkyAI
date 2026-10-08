//! Canvas REST client over reqwest. Read-only, one token per call.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;

use crate::core::traits::tools::canvas::Canvas;
use crate::core::types::tools::canvas::{
    Announcement, Assignment, AssignmentGrade, CalendarEvent, CanvasError, Course, CourseGrade,
};
use crate::runtime::tools::http;

/// Longest announcement body kept, in characters.
const BODY_CHARS: usize = 240;

/// text with HTML tags removed, whitespace collapsed, and held to BODY_CHARS.
fn plain(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_tag = false;
    for ch in text.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if in_tag => {}
            c if c.is_whitespace() => {
                if !out.ends_with(' ') {
                    out.push(' ');
                }
            }
            c => out.push(c),
        }
    }
    let trimmed = out.trim();
    if trimmed.chars().count() > BODY_CHARS {
        trimmed.chars().take(BODY_CHARS).collect::<String>() + "..."
    } else {
        trimmed.to_owned()
    }
}

/// A Canvas instance reached over HTTPS.
pub struct HttpCanvas {
    http: Client,
    base_url: String,
    page_size: usize,
}

/// A course as the courses endpoint returns it.
#[derive(Debug, Deserialize)]
struct RawCourse {
    id: u64,
    #[serde(default)]
    name: String,
    #[serde(default)]
    course_code: String,
    #[serde(default)]
    enrollments: Vec<RawEnrollment>,
}

/// The caller's enrollment in a course, present when total_scores is included.
#[derive(Debug, Deserialize)]
struct RawEnrollment {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    computed_current_score: Option<f64>,
    #[serde(default)]
    computed_current_grade: Option<String>,
}

/// One item from the self todo endpoint.
#[derive(Debug, Deserialize)]
struct RawTodo {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    context_name: String,
    assignment: Option<RawAssignment>,
}

/// The assignment carried by a todo item.
#[derive(Debug, Deserialize)]
struct RawAssignment {
    #[serde(default)]
    name: String,
    #[serde(default)]
    due_at: Option<DateTime<Utc>>,
    #[serde(default)]
    points_possible: Option<f64>,
    #[serde(default)]
    html_url: Option<String>,
}

/// One announcement from the announcements endpoint.
#[derive(Debug, Deserialize)]
struct RawAnnouncement {
    #[serde(default)]
    title: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    posted_at: Option<DateTime<Utc>>,
    #[serde(default)]
    html_url: Option<String>,
    #[serde(default)]
    context_code: String,
}

/// One item from the upcoming events endpoint.
#[derive(Debug, Deserialize)]
struct RawEvent {
    #[serde(default)]
    title: String,
    #[serde(default)]
    start_at: Option<DateTime<Utc>>,
    #[serde(default)]
    location_name: Option<String>,
    #[serde(default)]
    html_url: Option<String>,
}

/// One assignment with the caller's submission included.
#[derive(Debug, Deserialize)]
struct RawGradedAssignment {
    #[serde(default)]
    name: String,
    #[serde(default)]
    points_possible: Option<f64>,
    submission: Option<RawSubmission>,
}

/// The caller's submission on an assignment.
#[derive(Debug, Deserialize)]
struct RawSubmission {
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    grade: Option<String>,
}

impl HttpCanvas {
    /// Builds the client over the instance base URL, request budget, and page size.
    pub fn new(base_url: &str, timeout: Duration, page_size: usize) -> Result<Self, CanvasError> {
        let http = http::client(timeout)
            .map_err(|e| CanvasError::Unreachable(http::failure(&e).to_owned()))?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_owned(),
            page_size,
        })
    }

    /// One authenticated GET of a Canvas API path, deserialized into T.
    async fn get<T: for<'de> Deserialize<'de>>(
        &self,
        token: &SecretString,
        path: &str,
        query: &[(&str, String)],
    ) -> Result<T, CanvasError> {
        let url = reqwest::Url::parse_with_params(
            &format!("{}/api/v1/{path}", self.base_url),
            query.iter().map(|(k, v)| (*k, v.as_str())),
        )
        .map_err(|e| CanvasError::Unreachable(e.to_string()))?;
        let response = self
            .http
            .get(url)
            .bearer_auth(token.expose_secret())
            .send()
            .await
            .map_err(|e| CanvasError::Unreachable(http::failure(&e).to_owned()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(CanvasError::Refused(status.as_u16()));
        }
        response
            .json::<T>()
            .await
            .map_err(|e| CanvasError::Malformed(http::failure(&e).to_owned()))
    }
}

#[async_trait]
impl Canvas for HttpCanvas {
    async fn courses(&self, token: &SecretString) -> Result<Vec<Course>, CanvasError> {
        let raw: Vec<RawCourse> = self
            .get(
                token,
                "courses",
                &[
                    ("enrollment_state", "active".to_owned()),
                    ("per_page", self.page_size.to_string()),
                ],
            )
            .await?;
        Ok(raw
            .into_iter()
            .map(|c| Course {
                id: c.id,
                name: c.name,
                code: c.course_code,
            })
            .collect())
    }

    async fn assignments(&self, token: &SecretString) -> Result<Vec<Assignment>, CanvasError> {
        let raw: Vec<RawTodo> = self
            .get(
                token,
                "users/self/todo",
                &[("per_page", self.page_size.to_string())],
            )
            .await?;
        let mut out: Vec<Assignment> = raw
            .into_iter()
            .filter(|t| t.kind == "submitting")
            .filter_map(|t| {
                t.assignment.map(|a| Assignment {
                    name: a.name,
                    course: t.context_name,
                    due_at: a.due_at,
                    points: a.points_possible,
                    url: a.html_url,
                })
            })
            .collect();
        out.sort_by(|a, b| match (a.due_at, b.due_at) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });
        Ok(out)
    }

    async fn grades(&self, token: &SecretString) -> Result<Vec<CourseGrade>, CanvasError> {
        let raw: Vec<RawCourse> = self
            .get(
                token,
                "courses",
                &[
                    ("enrollment_state", "active".to_owned()),
                    ("include[]", "total_scores".to_owned()),
                    ("per_page", self.page_size.to_string()),
                ],
            )
            .await?;
        Ok(raw
            .into_iter()
            .map(|c| {
                let (score, grade) = match c
                    .enrollments
                    .into_iter()
                    .find(|e| e.kind == "student" || e.kind == "StudentEnrollment")
                {
                    Some(e) => (e.computed_current_score, e.computed_current_grade),
                    None => (None, None),
                };
                CourseGrade {
                    course: c.name,
                    score,
                    grade,
                }
            })
            .collect())
    }

    async fn announcements(&self, token: &SecretString) -> Result<Vec<Announcement>, CanvasError> {
        let courses = self.courses(token).await?;
        if courses.is_empty() {
            return Ok(Vec::new());
        }
        let names: HashMap<u64, String> = courses.iter().map(|c| (c.id, c.name.clone())).collect();
        let mut query = vec![
            ("per_page", self.page_size.to_string()),
            ("active_only", "true".to_owned()),
        ];
        for c in &courses {
            query.push(("context_codes[]", format!("course_{}", c.id)));
        }
        let raw: Vec<RawAnnouncement> = self.get(token, "announcements", &query).await?;
        Ok(raw
            .into_iter()
            .map(|a| {
                let course = a
                    .context_code
                    .strip_prefix("course_")
                    .and_then(|id| id.parse::<u64>().ok())
                    .and_then(|id| names.get(&id).cloned())
                    .unwrap_or_default();
                let body = if a.message.trim().is_empty() {
                    None
                } else {
                    Some(plain(&a.message))
                };
                Announcement {
                    title: a.title,
                    course,
                    posted_at: a.posted_at,
                    body,
                    url: a.html_url,
                }
            })
            .collect())
    }

    async fn calendar(&self, token: &SecretString) -> Result<Vec<CalendarEvent>, CanvasError> {
        let raw: Vec<RawEvent> = self.get(token, "users/self/upcoming_events", &[]).await?;
        Ok(raw
            .into_iter()
            .take(self.page_size)
            .map(|e| CalendarEvent {
                title: e.title,
                start_at: e.start_at,
                location: e.location_name,
                url: e.html_url,
            })
            .collect())
    }

    async fn assignment_grades(
        &self,
        token: &SecretString,
    ) -> Result<Vec<AssignmentGrade>, CanvasError> {
        let courses = self.courses(token).await?;
        let mut out: Vec<AssignmentGrade> = Vec::new();
        for course in courses.into_iter().take(self.page_size) {
            let raw: Vec<RawGradedAssignment> = self
                .get(
                    token,
                    &format!("courses/{}/assignments", course.id),
                    &[
                        ("include[]", "submission".to_owned()),
                        ("per_page", self.page_size.to_string()),
                    ],
                )
                .await?;
            for a in raw {
                let Some(sub) = a.submission else { continue };
                if sub.score.is_none() && sub.grade.is_none() {
                    continue;
                }
                out.push(AssignmentGrade {
                    course: course.name.clone(),
                    name: a.name,
                    score: sub.score,
                    points: a.points_possible,
                    grade: sub.grade,
                });
            }
            if out.len() >= self.page_size {
                break;
            }
        }
        out.truncate(self.page_size);
        Ok(out)
    }
}
