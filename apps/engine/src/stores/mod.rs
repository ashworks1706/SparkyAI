//! Store adapters. The only place a database connection or a platform client is opened.
//!
//! Imports only core. standalone holds the self-hosted adapters; platform holds the HTTP ones.

pub mod platform;
#[cfg(feature = "standalone")]
pub mod standalone;
