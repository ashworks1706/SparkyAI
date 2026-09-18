//! A building or place on the ASU campus map, with a map link.

use super::{LiveSource, Param};

/// The ASU campus map.
pub struct CampusMap;

const PARAMS: &[Param] = &[Param::text("place", "Building name, code or place.")
    .required()
    .example("Hayden")];

impl LiveSource for CampusMap {
    fn key(&self) -> &'static str {
        "campus_map"
    }

    fn hint(&self) -> &'static str {
        "where a building or place is"
    }

    fn label(&self) -> &'static str {
        "ASU Campus Map"
    }

    fn category(&self) -> &'static str {
        "campus"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
