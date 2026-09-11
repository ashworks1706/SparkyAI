//! Store adapters. The only place a database connection is opened.
//! Implements the core::traits store traits; imports only core.

pub mod confirmation;
pub mod conversation;
pub mod knowledge;
pub mod memory;
pub mod postgres;
