//! Shared foundations: configuration, telemetry, the owned domain types, and the test suite.
//! Imports nothing else in this crate; everything else may import it.

pub mod config;
pub mod telemetry;
pub mod types;

#[cfg(test)]
pub mod tests;
