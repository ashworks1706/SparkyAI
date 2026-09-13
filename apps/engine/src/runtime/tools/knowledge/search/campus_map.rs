//! search_campus_map: a building or place on the ASU campus map, with a map link.

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

    fn description(&self) -> &'static str {
        "Find a building or place on the ASU campus map: what it is and a map link."
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
