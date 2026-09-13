//! search_study_rooms: bookable study room slots at one ASU library on one date.

use super::{LiveSource, Param};

/// LibCal study room availability.
pub struct StudyRooms;

const LIBRARIES: &[&str] = &["hayden", "noble", "fletcher", "west", "polytechnic"];

const PARAMS: &[Param] = &[
    Param::one_of("library", "Library.", LIBRARIES).required(),
    Param::date("date", "Date to check.")
        .required()
        .example("2026-09-14"),
];

impl LiveSource for StudyRooms {
    fn key(&self) -> &'static str {
        "study_rooms"
    }

    fn description(&self) -> &'static str {
        "List open study room slots at one ASU library on one date."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
