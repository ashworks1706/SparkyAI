//! search_sports: the Sun Devil schedule and recent results for one sport.

use super::{LiveSource, Param};

/// Sun Devil Athletics schedules.
pub struct Sports;

const SPORTS: &[&str] = &[
    "football",
    "men's basketball",
    "women's basketball",
    "baseball",
    "softball",
    "volleyball",
    "soccer",
    "hockey",
    "wrestling",
];

const PARAMS: &[Param] = &[Param::one_of("sport", "Sport.", SPORTS).required()];

impl LiveSource for Sports {
    fn key(&self) -> &'static str {
        "sports"
    }

    fn description(&self) -> &'static str {
        "Fetch the Sun Devil schedule and recent results for one sport."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
