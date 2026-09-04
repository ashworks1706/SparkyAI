//! Agent runtime: the traits every adapter implements, the default policy, the trace sinks,
//! context assembly, and the loop. Types live in `core::types`; this module imports only `core`.

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
