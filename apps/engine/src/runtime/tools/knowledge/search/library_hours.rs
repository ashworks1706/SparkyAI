//! This week's opening hours at every ASU library.

use super::{LiveSource, Param};

/// The ASU Library hours page.
pub struct LibraryHours;

impl LiveSource for LibraryHours {
    fn key(&self) -> &'static str {
        "library_hours"
    }

    fn hint(&self) -> &'static str {
        "library opening hours"
    }

    fn label(&self) -> &'static str {
        "ASU Library Hours"
    }

    fn category(&self) -> &'static str {
        "library"
    }

    fn params(&self) -> &'static [Param] {
        &[]
    }
}
