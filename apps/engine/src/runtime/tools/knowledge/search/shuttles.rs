//! Live next-bus times at every stop of the ASU intercampus shuttles.

use super::{Freshness, LiveSource, Param};

/// The ASU shuttle tracker.
pub struct Shuttles;

const ROUTES: &[&str] = &[
    "mercado",
    "polytechnic-tempe",
    "tempe-downtown phoenix-west",
    "tempe-west express",
];

const PARAMS: &[Param] = &[Param::one_of(
    "route",
    "Shuttle route. Leave out for every route.",
    ROUTES,
)];

impl LiveSource for Shuttles {
    fn key(&self) -> &'static str {
        "shuttles"
    }

    fn hint(&self) -> &'static str {
        "intercampus shuttle times"
    }

    fn label(&self) -> &'static str {
        "ASU Shuttle Tracker"
    }

    fn category(&self) -> &'static str {
        "transit"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }

    fn freshness(&self) -> Freshness {
        Freshness::Live
    }
}
