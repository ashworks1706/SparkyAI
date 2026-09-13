//! search_courses: sections, instructors, meeting days and open seats for one term.

use serde_json::{Map, Value};

use super::{LiveSource, Param};

/// The ASU class search.
pub struct Courses;

const LEVELS: &[&str] = &[
    "lower division",
    "upper division",
    "undergraduate",
    "graduate",
    "100-199",
    "200-299",
    "300-399",
    "400-499",
];

const DAYS: &[&str] = &[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

const SESSIONS: &[&str] = &["a", "b", "c", "other"];

const SEASONS: &[&str] = &["spring", "summer", "fall"];

const PARAMS: &[Param] = &[
    Param::text("term", "Term to search: spring, summer or fall and a year.")
        .required()
        .example("Fall 2026"),
    Param::text("keywords", "Subject, course number or title.").example("CSE 310"),
    Param::any_of("level", "Course levels.", LEVELS),
    Param::any_of("days", "Meeting days.", DAYS),
    Param::one_of("session", "Session within the term.", SESSIONS),
    Param::flag("open_only", "Only sections with open seats."),
];

impl LiveSource for Courses {
    fn key(&self) -> &'static str {
        "courses"
    }

    fn description(&self) -> &'static str {
        "Search the ASU class catalog for one term: sections, instructors, meeting days, and \
         open seats."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }

    fn check(&self, params: &Map<String, Value>) -> Result<(), String> {
        let term = params
            .get("term")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let parts: Vec<&str> = term.split_whitespace().collect();
        let season_ok = parts
            .first()
            .is_some_and(|s| SEASONS.iter().any(|k| k.eq_ignore_ascii_case(s)));
        let year_ok = parts
            .get(1)
            .is_some_and(|y| y.len() == 4 && y.chars().all(|c| c.is_ascii_digit()));
        if parts.len() == 2 && season_ok && year_ok {
            Ok(())
        } else {
            Err(format!("term must look like Fall 2026, got {term:?}"))
        }
    }
}
