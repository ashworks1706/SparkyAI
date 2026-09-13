//! search_library_hours: this week's opening hours at every ASU library.

use super::{LiveSource, Param};

/// The ASU Library hours page.
pub struct LibraryHours;

impl LiveSource for LibraryHours {
    fn key(&self) -> &'static str {
        "library_hours"
    }

    fn description(&self) -> &'static str {
        "Fetch this week's opening hours for every ASU library."
    }

    fn params(&self) -> &'static [Param] {
        &[]
    }
}
