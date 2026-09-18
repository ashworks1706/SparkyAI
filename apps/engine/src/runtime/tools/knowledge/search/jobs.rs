//! ASU student employment listings and how to apply.

use super::{LiveSource, Param};

/// The ASU student employment page.
pub struct Jobs;

impl LiveSource for Jobs {
    fn key(&self) -> &'static str {
        "jobs"
    }

    fn hint(&self) -> &'static str {
        "student employment listings"
    }

    fn label(&self) -> &'static str {
        "ASU Student Employment"
    }

    fn category(&self) -> &'static str {
        "jobs"
    }

    fn params(&self) -> &'static [Param] {
        &[]
    }
}
