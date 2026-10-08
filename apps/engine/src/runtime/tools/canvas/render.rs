//! Rendering of the Canvas replies: courses, assignments, grades, announcements, calendar.

use std::fmt::Write as _;

use crate::core::types::tools::ToolOutput;
use crate::core::types::tools::canvas::{
    Announcement, Assignment, AssignmentGrade, CalendarEvent, Course, CourseGrade,
};
use crate::runtime::tools::structured;

/// A due date as the model reads it, or a note that none is set.
fn due(assignment: &Assignment) -> String {
    match assignment.due_at {
        Some(at) => at.format("%b %d, %Y %H:%M UTC").to_string(),
        None => "no due date".to_owned(),
    }
}

/// The courses reply.
pub(super) fn courses_output(courses: Vec<Course>, max: usize) -> ToolOutput {
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
pub(super) fn assignments_output(assignments: Vec<Assignment>, max: usize) -> ToolOutput {
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
pub(super) fn grades_output(grades: Vec<CourseGrade>, max: usize) -> ToolOutput {
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

/// A moment as the model reads it, or a note that none is set.
fn moment(at: Option<chrono::DateTime<chrono::Utc>>) -> String {
    match at {
        Some(at) => at.format("%b %d, %Y %H:%M UTC").to_string(),
        None => "no date".to_owned(),
    }
}

/// The announcements reply.
pub(super) fn announcements_output(items: Vec<Announcement>, max: usize) -> ToolOutput {
    let shown: Vec<Announcement> = items.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No recent announcements.".to_owned()
    } else {
        let mut lines = format!("{} recent announcements:", shown.len());
        for a in &shown {
            let body = a
                .body
                .as_deref()
                .map(|b| format!(" - {b}"))
                .unwrap_or_default();
            let link = a
                .url
                .as_deref()
                .map(|u| format!(" {u}"))
                .unwrap_or_default();
            let _ = write!(
                lines,
                "\n- {} ({}) {}{}{}",
                a.title,
                a.course,
                moment(a.posted_at),
                body,
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

/// The calendar reply.
pub(super) fn calendar_output(items: Vec<CalendarEvent>, max: usize) -> ToolOutput {
    let shown: Vec<CalendarEvent> = items.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No upcoming calendar events.".to_owned()
    } else {
        let mut lines = format!("{} upcoming events:", shown.len());
        for e in &shown {
            let place = e
                .location
                .as_deref()
                .map(|l| format!(" at {l}"))
                .unwrap_or_default();
            let link = e
                .url
                .as_deref()
                .map(|u| format!(" {u}"))
                .unwrap_or_default();
            let _ = write!(
                lines,
                "\n- {} {}{}{}",
                e.title,
                moment(e.start_at),
                place,
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

/// The per-assignment grades reply.
pub(super) fn assignment_grades_output(items: Vec<AssignmentGrade>, max: usize) -> ToolOutput {
    let shown: Vec<AssignmentGrade> = items.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No graded assignments yet.".to_owned()
    } else {
        let mut lines = "Assignment grades:".to_owned();
        for g in &shown {
            let mark = match (&g.grade, g.score, g.points) {
                (Some(letter), Some(score), Some(points)) => format!("{letter} ({score}/{points})"),
                (_, Some(score), Some(points)) => format!("{score}/{points}"),
                (Some(letter), _, _) => letter.clone(),
                (_, Some(score), None) => score.to_string(),
                _ => "graded".to_owned(),
            };
            let _ = write!(lines, "\n- {} ({}): {}", g.name, g.course, mark);
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}
