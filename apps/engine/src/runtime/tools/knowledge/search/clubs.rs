//! search_clubs: student organizations on Sun Devil Central.

use super::{LiveSource, Param};

/// The Sun Devil Central group directory.
pub struct Clubs;

const PARAMS: &[Param] = &[Param::text("keywords", "Club name or topic.")
    .required()
    .example("artificial intelligence")];

impl LiveSource for Clubs {
    fn key(&self) -> &'static str {
        "clubs"
    }

    fn description(&self) -> &'static str {
        "Search ASU student organizations and clubs on Sun Devil Central."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
