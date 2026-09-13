//! search_sports_news: Sun Devil Athletics news and outside coverage, for one sport or all.

use super::{LiveSource, Param};

/// Sun Devil Athletics news feeds.
pub struct SportsNews;

const SPORTS: &[&str] = &[
    "football",
    "men's basketball",
    "women's basketball",
    "baseball",
    "softball",
    "volleyball",
    "beach volleyball",
    "soccer",
    "ice hockey",
    "wrestling",
    "gymnastics",
    "men's golf",
    "women's golf",
    "cross country",
    "track and field",
    "triathlon",
    "men's tennis",
    "women's tennis",
    "men's swimming and diving",
    "women's swimming and diving",
];

const PARAMS: &[Param] = &[
    Param::one_of("sport", "Sport. Leave out for every sport.", SPORTS),
    Param::text("keywords", "Words in the headline.").example("Texas A&M"),
];

impl LiveSource for SportsNews {
    fn key(&self) -> &'static str {
        "sports_news"
    }

    fn description(&self) -> &'static str {
        "Latest Sun Devil Athletics news, and outside coverage, for one sport or all."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
