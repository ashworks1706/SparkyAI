//! The Sun Devil schedule and recent results for one sport.

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

    fn hint(&self) -> &'static str {
        "Sun Devil schedules and results"
    }

    fn label(&self) -> &'static str {
        "Sun Devil Athletics"
    }

    fn category(&self) -> &'static str {
        "sports"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
