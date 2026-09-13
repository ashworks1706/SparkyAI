//! search_web: web results from Google, Brave and Bing through self-hosted SearXNG.

use super::{LiveSource, Param};

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

    fn description(&self) -> &'static str {
        "Search the web through Google, Brave and Bing for anything the ASU sources do not cover."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
