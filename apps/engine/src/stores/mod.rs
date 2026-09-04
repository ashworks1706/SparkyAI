//! Store adapters. The only place a database connection is opened.
//! Implements the `agent::harness` store traits; imports only `core` and `agent::harness`.

pub mod postgres;
