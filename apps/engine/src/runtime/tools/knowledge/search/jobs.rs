//! search_jobs: ASU student employment listings and how to apply.

use super::{LiveSource, Param};

/// The ASU student employment page.
pub struct Jobs;

impl LiveSource for Jobs {
    fn key(&self) -> &'static str {
        "jobs"
    }

    fn description(&self) -> &'static str {
        "Fetch ASU student employment listings and how to apply."
    }

    fn params(&self) -> &'static [Param] {
        &[]
    }
}
