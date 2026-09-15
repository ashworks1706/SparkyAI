//! Upcoming events on the ASU events calendar.

use super::{LiveSource, Param};

/// The ASU events calendar.
pub struct Events;

const PARAMS: &[Param] =
    &[Param::text("keywords", "What the event is about.").example("career fair")];

impl LiveSource for Events {
    fn key(&self) -> &'static str {
        "events"
    }

    fn hint(&self) -> &'static str {
        "the events calendar"
    }

    fn category(&self) -> &'static str {
        "events"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
