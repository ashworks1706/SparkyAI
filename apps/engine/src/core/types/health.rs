//! Readiness report returned by `/health/ready`.

use serde::Serialize;

/// Which dependencies answered.
#[derive(Debug, Serialize)]
pub struct Readiness {
    /// Postgres answered `select 1`.
    pub postgres: bool,
    /// The model endpoint listed its models.
    pub model: bool,
}
