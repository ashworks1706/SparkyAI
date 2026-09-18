//! Web results from Google, Brave and Bing through self-hosted SearXNG.

use super::{Freshness, LiveSource, Param};

/// Web search.
pub struct Web;

const TIME_RANGES: &[&str] = &["day", "week", "month", "year"];

const PARAMS: &[Param] = &[
    Param::text("query", "What to search for.")
        .required()
        .example("ASU vs Texas A&M score"),
    Param::one_of(
        "time_range",
        "Only results from this recent period.",
        TIME_RANGES,
    ),
];

impl LiveSource for Web {
    fn key(&self) -> &'static str {
        "web"
    }

    fn hint(&self) -> &'static str {
        "the open web, when no ASU source fits"
    }

    fn label(&self) -> &'static str {
        "Web Search"
    }

    fn category(&self) -> &'static str {
        "web"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }

    fn freshness(&self) -> Freshness {
        Freshness::Live
    }
}
