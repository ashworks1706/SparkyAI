//! Sections, instructors, meeting days and open seats for one term.

use chrono::{Datelike, NaiveDate};
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

/// The term running on date: spring through April, summer through July, fall after it.
pub fn term_for(date: NaiveDate) -> String {
    let season = match date.month() {
        1..=4 => "Spring",
        5..=7 => "Summer",
        _ => "Fall",
    };
    format!("{season} {}", date.year())
}

/// The term query names, as the catalog spells it.
pub fn term_in(query: &str) -> Option<String> {
    let words: Vec<&str> = query.split_whitespace().collect();
    words.windows(2).find_map(|pair| {
        let season = SEASONS.iter().find(|s| {
            s.eq_ignore_ascii_case(pair[0].trim_matches(|c: char| !c.is_alphanumeric()))
        })?;
        let year: String = pair[1].chars().filter(char::is_ascii_digit).collect();
        (year.len() == 4).then(|| format!("{}{} {year}", season[..1].to_uppercase(), &season[1..]))
    })
}

/// Words a student writes around a course. The class search keyword box matches subject,
/// catalog number and course title.
const INTENT_WORDS: &[&str] = &[
    "about",
    "any",
    "are",
    "availability",
    "available",
    "class",
    "classes",
    "course",
    "courses",
    "credit",
    "credits",
    "description",
    "do",
    "does",
    "enrol",
    "enroll",
    "enrollment",
    "full",
    "get",
    "have",
    "hours",
    "instructor",
    "is",
    "left",
    "many",
    "me",
    "meet",
    "meeting",
    "offered",
    "open",
    "prereq",
    "prereqs",
    "prerequisite",
    "prerequisites",
    "professor",
    "requirement",
    "requirements",
    "room",
    "schedule",
    "seat",
    "seats",
    "section",
    "sections",
    "spots",
    "syllabus",
    "take",
    "teaches",
    "this",
    "time",
    "times",
    "what",
    "when",
    "where",
    "which",
    "who",
];

/// Short English words that a course title carries, so narrow keeps them.
const CONNECTIVES: &[&str] = &["a", "an", "and", "for", "in", "of", "the", "to", "with"];

/// Whether word is one a student writes rather than a subject the catalog knows.
fn written_around(word: &str) -> bool {
    let lower = word.to_lowercase();
    INTENT_WORDS.contains(&lower.as_str()) || CONNECTIVES.contains(&lower.as_str())
}

/// Whether word is a subject code: two to four letters, as a catalog subject is written.
fn subject_like(word: &str) -> bool {
    (2..=4).contains(&word.len()) && word.chars().all(|c| c.is_ascii_alphabetic())
}

/// Whether word is a catalog number: three digits, with an optional trailing letter.
fn number_like(word: &str) -> bool {
    let digits = word.trim_end_matches(char::is_alphabetic);
    digits.len() == 3
        && digits.chars().all(|c| c.is_ascii_digit())
        && word.len() - digits.len() <= 1
}

/// The course code query names, like CSE 310, written as the catalog spells it.
pub fn code_in(query: &str) -> Option<String> {
    let words: Vec<&str> = query
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let joined = words.iter().find_map(|word| {
        let split = word.find(|c: char| c.is_ascii_digit())?;
        let (subject, number) = word.split_at(split);
        (subject_like(subject) && number_like(number))
            .then(|| format!("{} {number}", subject.to_uppercase()))
    });
    joined.or_else(|| {
        words.windows(2).find_map(|pair| {
            (subject_like(pair[0]) && !written_around(pair[0]) && number_like(pair[1]))
                .then(|| format!("{} {}", pair[0].to_uppercase(), pair[1]))
        })
    })
}

/// Whether word is one the class search keyword box can match.
fn matchable(word: &str) -> bool {
    let bare = word
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if bare.is_empty() || INTENT_WORDS.contains(&bare.as_str()) {
        return false;
    }
    if SEASONS.iter().any(|s| s.eq_ignore_ascii_case(&bare)) {
        return false;
    }
    // A four digit number is a year, which the term parameter carries.
    !(bare.len() == 4 && bare.chars().all(|c| c.is_ascii_digit()))
}

/// Whether query asks for sections that still have room.
fn asks_open(query: &str) -> bool {
    let lower = query.to_lowercase();
    [
        "open seat",
        "open section",
        "seats left",
        "still open",
        "spots left",
        "space left",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase))
}

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

    fn hint(&self) -> &'static str {
        "class sections, instructors, open seats"
    }

    fn label(&self) -> &'static str {
        "ASU Class Search"
    }

    fn category(&self) -> &'static str {
        "courses"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }

    fn query_param(&self) -> Option<&'static str> {
        Some("keywords")
    }

    fn narrow(&self, query: &str) -> String {
        if let Some(code) = code_in(query) {
            return code;
        }
        let kept: Vec<&str> = query.split_whitespace().filter(|w| matchable(w)).collect();
        if kept.is_empty() {
            return query.trim().to_owned();
        }
        kept.join(" ")
    }

    fn derived(&self, query: &str, today: NaiveDate) -> Vec<(&'static str, String)> {
        let mut out = vec![("term", term_in(query).unwrap_or_else(|| term_for(today)))];
        if asks_open(query) {
            out.push(("open_only", "true".to_owned()));
        }
        out
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
