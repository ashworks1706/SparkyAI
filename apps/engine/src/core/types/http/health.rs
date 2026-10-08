//! Readiness report returned by /health/ready.

use std::collections::BTreeMap;

use serde::Serialize;

/// Which dependencies answered, each under its probe name.
#[derive(Debug, Serialize)]
pub struct Readiness {
    /// Each store probe by name, for example postgres or platform.
    #[serde(flatten)]
    pub stores: BTreeMap<&'static str, bool>,
    /// The model endpoint listed its models.
    pub model: bool,
}
