//! Store adapters. The only place a database connection is opened.
//! Implements the core::traits store traits; imports only core.

pub mod postgres;
pub mod profile;
pub mod skills;
