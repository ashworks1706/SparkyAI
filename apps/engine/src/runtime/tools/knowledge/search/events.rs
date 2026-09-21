//! Upcoming events on the ASU events calendar.

use super::{Freshness, LiveSource, Param};

/// The ASU events calendar.
pub struct Events;

const PARAMS: &[Param] =
    &[Param::text("keywords", "What the event is about.").example("career fair")];

impl LiveSource for Events {
    fn key(&self) -> &'static str {
        "events"
    }

    fn hint(&self) -> &'static str {
        "the events calendar and Sun Devil Central listings"
    }

    fn label(&self) -> &'static str {
        "ASU Events"
    }

    fn category(&self) -> &'static str {
        "events"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }

    fn freshness(&self) -> Freshness {
        // Merges the login-gated Sun Devil Central listings, so its answers are never indexed.
        Freshness::Live
    }
}
