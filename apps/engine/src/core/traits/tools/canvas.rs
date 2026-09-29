//! Canvas LMS read API.

use async_trait::async_trait;
use secrecy::SecretString;

use crate::core::types::tools::canvas::{
    Announcement, Assignment, AssignmentGrade, CalendarEvent, CanvasError, Course, CourseGrade,
};

/// Read-only access to a Canvas instance on behalf of one token holder.
#[async_trait]
pub trait Canvas: Send + Sync {
    /// The active courses the token holder is enrolled in.
    async fn courses(&self, token: &SecretString) -> Result<Vec<Course>, CanvasError>;

    /// The assignments the token holder still has to submit, soonest first.
    async fn assignments(&self, token: &SecretString) -> Result<Vec<Assignment>, CanvasError>;

    /// The token holder's current grade in each active course.
    async fn grades(&self, token: &SecretString) -> Result<Vec<CourseGrade>, CanvasError>;

    /// Recent announcements across the token holder's active courses.
    async fn announcements(&self, token: &SecretString) -> Result<Vec<Announcement>, CanvasError>;

    /// The token holder's upcoming calendar items.
    async fn calendar(&self, token: &SecretString) -> Result<Vec<CalendarEvent>, CanvasError>;

    /// The token holder's grade on each graded assignment.
    async fn assignment_grades(
        &self,
        token: &SecretString,
    ) -> Result<Vec<AssignmentGrade>, CanvasError>;
}
