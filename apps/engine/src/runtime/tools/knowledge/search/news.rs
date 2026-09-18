//! Recent stories on ASU News.

use super::{LiveSource, Param};

/// The ASU News search.
pub struct News;

const PARAMS: &[Param] = &[Param::text(
    "keywords",
    "Topic to search for. Leave out for the newest stories.",
)
.example("robotics")];

impl LiveSource for News {
    fn key(&self) -> &'static str {
        "news"
    }

    fn hint(&self) -> &'static str {
        "ASU News stories"
    }

    fn label(&self) -> &'static str {
        "ASU News"
    }

    fn category(&self) -> &'static str {
        "news"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
