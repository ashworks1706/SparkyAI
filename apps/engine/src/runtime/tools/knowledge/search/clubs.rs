//! Student organizations on Sun Devil Central.

use super::{Freshness, LiveSource, Param};

/// The Sun Devil Central group directory.
pub struct Clubs;

const PARAMS: &[Param] = &[Param::text("keywords", "Club name or topic.")
    .required()
    .example("artificial intelligence")];

impl LiveSource for Clubs {
    fn key(&self) -> &'static str {
        "clubs"
    }

    fn hint(&self) -> &'static str {
        "student organizations"
    }

    fn label(&self) -> &'static str {
        "Sun Devil Central"
    }

    fn category(&self) -> &'static str {
        "clubs"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }

    fn freshness(&self) -> Freshness {
        // Login-gated and read through the admin session, so its answers are never indexed.
        Freshness::Live
    }
}
