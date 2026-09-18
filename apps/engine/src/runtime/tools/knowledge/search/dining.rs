//! Dining hall and campus restaurant hours for one campus.

use super::{LiveSource, Param};

/// The Sun Devil Hospitality hours of one campus.
pub struct Dining;

const CAMPUSES: &[&str] = &["tempe", "downtown", "west"];

const PARAMS: &[Param] = &[
    Param::one_of("campus", "Campus whose dining hours to read.", CAMPUSES)
        .required()
        .example("tempe"),
];

impl LiveSource for Dining {
    fn key(&self) -> &'static str {
        "dining"
    }

    fn hint(&self) -> &'static str {
        "dining hall and campus restaurant hours"
    }

    fn label(&self) -> &'static str {
        "Sun Devil Dining"
    }

    fn category(&self) -> &'static str {
        "dining"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
