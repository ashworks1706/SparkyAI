//! search_live_shuttles: live next-bus times at every stop of the ASU intercampus shuttles.

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

    fn description(&self) -> &'static str {
        "Live next-bus times at every stop of the ASU intercampus shuttles."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }

    fn freshness(&self) -> Freshness {
        Freshness::Live
    }
}
