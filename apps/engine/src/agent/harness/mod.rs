//! Agent runtime: traits, loop, context assembly, tracing.
//! Imports only `core`; nothing else in this crate.

pub mod agent;
pub mod assemble;
pub mod conversation;
pub mod memory;
pub mod model;
pub mod policy;
pub mod retrieval;
pub mod sandbox;
pub mod tool;
pub mod trace;
