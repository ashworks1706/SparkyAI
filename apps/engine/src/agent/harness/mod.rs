//! Agent runtime behaviour: the loop, context assembly, the tool registry, the default policy,
//! and the trace sinks. Types are in `core::types`, interfaces in `core::traits`; this module
//! imports only `core`.

pub mod agent;
pub mod assemble;
pub mod policy;
pub mod tool;
pub mod trace;
