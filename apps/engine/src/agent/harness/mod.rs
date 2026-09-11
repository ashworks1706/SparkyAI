//! Agent runtime behaviour: the loop, context assembly, the tool registry, the default policy,
//! and the trace sinks. Types are in core::types, interfaces in core::traits. This module
//! imports only core.

pub mod agent;
pub mod compact;
pub mod memory;
pub mod safety;
pub mod tools;
pub mod trace;
