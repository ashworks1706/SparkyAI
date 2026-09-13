//! search_events: upcoming events on the ASU events calendar.

use super::{LiveSource, Param};

/// The ASU events calendar.
pub struct Events;

const PARAMS: &[Param] =
    &[Param::text("keywords", "What the event is about.").example("career fair")];

impl LiveSource for Events {
    fn key(&self) -> &'static str {
        "events"
    }

    fn description(&self) -> &'static str {
        "Search the ASU events calendar for upcoming events."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
