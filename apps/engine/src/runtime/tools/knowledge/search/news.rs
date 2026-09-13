//! search_news: recent stories on ASU News.

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

    fn description(&self) -> &'static str {
        "The newest ASU News stories, or a search of ASU News by topic."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
