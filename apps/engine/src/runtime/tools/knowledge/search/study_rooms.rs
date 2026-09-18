//! Bookable study room slots at one ASU library on one date.

use super::{Freshness, LiveSource, Param};

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

    fn hint(&self) -> &'static str {
        "bookable library study rooms"
    }

    fn label(&self) -> &'static str {
        "ASU Library Study Rooms"
    }

    fn category(&self) -> &'static str {
        "library"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }

    fn freshness(&self) -> Freshness {
        Freshness::Live
    }
}
