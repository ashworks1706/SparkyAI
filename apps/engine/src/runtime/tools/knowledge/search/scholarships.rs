//! ASU scholarships by keyword, citizenship, year of study and field.

use super::{LiveSource, Param};

/// The ASU scholarship search.
pub struct Scholarships;

const CITIZENSHIP: &[&str] = &[
    "us citizen",
    "us permanent resident",
    "daca/dreamer",
    "international student",
];

const APPLICANTS: &[&str] = &[
    "first-year undergrad",
    "second-year undergrad",
    "third-year undergrad",
    "fourth-year+ undergrad",
    "graduate student",
    "undergraduate alumni",
    "graduate alumni",
];

const FOCUS: &[&str] = &[
    "business and entrepreneurship",
    "creative and performing arts",
    "environment and sustainability",
    "health and medicine",
    "humanities",
    "international affairs",
    "journalism and media",
    "national security",
    "public policy",
    "public service",
    "social justice",
    "social science",
    "stem",
    "peace and conflict resolution",
];

const PARAMS: &[Param] = &[
    Param::text("keywords", "Words in the scholarship name or description.").example("women"),
    Param::one_of("citizenship", "Citizenship status.", CITIZENSHIP),
    Param::one_of("applicant", "Year of study.", APPLICANTS),
    Param::one_of("focus", "Field of focus.", FOCUS),
];

impl LiveSource for Scholarships {
    fn key(&self) -> &'static str {
        "scholarships"
    }

    fn hint(&self) -> &'static str {
        "scholarships and who may apply"
    }

    fn label(&self) -> &'static str {
        "ASU Scholarship Search"
    }

    fn category(&self) -> &'static str {
        "scholarships"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
