//! Books, articles, journals and media in the ASU Library.

use super::{LiveSource, Param};

/// The ASU Library catalog.
pub struct LibraryCatalog;

const TYPES: &[&str] = &[
    "all",
    "books",
    "articles",
    "journals",
    "images",
    "scores",
    "maps",
    "sound recordings",
    "video",
];

const PARAMS: &[Param] = &[
    Param::text("keywords", "Title, author or subject.")
        .required()
        .example("deep learning"),
    Param::one_of("type", "Resource type.", TYPES),
];

impl LiveSource for LibraryCatalog {
    fn key(&self) -> &'static str {
        "library_catalog"
    }

    fn hint(&self) -> &'static str {
        "books, articles and media"
    }

    fn category(&self) -> &'static str {
        "library"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
