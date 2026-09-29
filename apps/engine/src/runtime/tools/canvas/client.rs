//! Canvas REST client over reqwest. Read-only, one token per call.

use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;

use crate::core::traits::tools::canvas::Canvas;
use crate::core::types::tools::canvas::{Assignment, CanvasError, Course, CourseGrade};

/// Names a reqwest failure without repeating the URL or the token.
fn kind_of(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        "timed out".to_owned()
    } else if error.is_connect() {
        "could not connect".to_owned()
    } else {
        "request failed".to_owned()
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

impl HttpCanvas {
    /// Builds the client over the instance base URL, request budget, and page size.
    pub fn new(base_url: &str, timeout: Duration, page_size: usize) -> Result<Self, CanvasError> {
        let http = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| CanvasError::Unreachable(kind_of(&e)))?;
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
            .map_err(|e| CanvasError::Unreachable(kind_of(&e)))?;
        let status = response.status();
        if !status.is_success() {
            return Err(CanvasError::Refused(status.as_u16()));
        }
        response
            .json::<T>()
            .await
            .map_err(|e| CanvasError::Malformed(kind_of(&e)))
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
}
