//! What one course covers, its credit hours, and the prerequisites it lists.

use chrono::NaiveDate;

use super::courses::code_in;
use super::{LiveSource, Param};

/// The ASU course catalog.
pub struct CourseCatalog;

const PARAMS: &[Param] = &[
    Param::text("keywords", "Course code, subject or title.")
        .required()
        .example("CSE 485"),
    Param::text("term", "Term the catalog entry is read for.").example("Fall 2026"),
];

impl LiveSource for CourseCatalog {
    fn key(&self) -> &'static str {
        "course_catalog"
    }

    fn hint(&self) -> &'static str {
        "what a course covers, its credits and prerequisites"
    }

    fn label(&self) -> &'static str {
        "ASU Course Catalog"
    }

    fn category(&self) -> &'static str {
        "course_catalog"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }

    fn query_param(&self) -> Option<&'static str> {
        Some("keywords")
    }

    fn narrow(&self, query: &str) -> String {
        code_in(query).unwrap_or_else(|| query.trim().to_owned())
    }

    fn derived(&self, query: &str, _today: NaiveDate) -> Vec<(&'static str, String)> {
        match super::courses::term_in(query) {
            Some(term) => vec![("term", term)],
            None => Vec::new(),
        }
    }
}
