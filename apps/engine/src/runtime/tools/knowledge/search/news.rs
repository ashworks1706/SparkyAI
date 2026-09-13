//! search_news: recent stories on ASU News.

use super::{LiveSource, Param};

/// The ASU News search.
pub struct News;

const PARAMS: &[Param] = &[Param::text("keywords", "What the story is about.")
    .required()
    .example("robotics")];

impl LiveSource for News {
    fn key(&self) -> &'static str {
        "news"
    }

    fn description(&self) -> &'static str {
        "Search ASU News for recent stories."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
