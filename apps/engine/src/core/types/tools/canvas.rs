//! Canvas read models and the errors a Canvas call returns.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One course the caller is enrolled in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Course {
    /// Canvas course id.
    pub id: u64,
    /// Course name.
    pub name: String,
    /// Course code, as the catalog lists it.
    pub code: String,
}

/// One assignment the caller still has to submit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    /// Assignment name.
    pub name: String,
    /// Course the assignment belongs to.
    pub course: String,
    /// When it is due, if a due date is set.
    pub due_at: Option<DateTime<Utc>>,
    /// Points it is worth, if set.
    pub points: Option<f64>,
    /// Link to the assignment.
    pub url: Option<String>,
}

/// The caller's current standing in one course.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CourseGrade {
    /// Course the grade is for.
    pub course: String,
    /// Current numeric score out of 100, if Canvas computes one.
    pub score: Option<f64>,
    /// Current letter grade, if the course uses one.
    pub grade: Option<String>,
}

/// A Canvas request that could not be answered.
#[derive(Debug, thiserror::Error)]
pub enum CanvasError {
    /// The Canvas host could not be reached.
    #[error("canvas is unreachable: {0}")]
    Unreachable(String),
    /// Canvas answered with an error status.
    #[error("canvas returned {0}")]
    Refused(u16),
    /// Canvas answered with something other than the expected shape.
    #[error("canvas returned an unexpected response: {0}")]
    Malformed(String),
}
