//! Shared foundations: configuration, telemetry, every type, every trait, and the test suite.
//! Imports nothing else in this crate; everything else may import it.

pub mod config;
pub mod telemetry;
pub mod traits;
pub mod types;

#[cfg(test)]
pub mod tests;
