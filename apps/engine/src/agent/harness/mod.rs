//! Agent runtime behaviour: the loop, context assembly, the tool registry, the default policy,
//! and the trace sinks. Types are in core::types, interfaces in core::traits. This module
//! imports only core.

pub mod agent;
pub mod assemble;
pub mod capability;
pub mod compact;
pub mod detect;
pub mod guardrail;
pub mod policy;
pub mod profile;
pub mod task;
pub mod tool;
pub mod trace;
