//! Sparky Harness — model-independent agent runtime.
//!
//! Types, traits, the agent loop, context assembly, and tracing.
//! See `docs/ARCHITECTURE.md`.

pub mod agent;
pub mod assemble;
pub mod context;
pub mod conversation;
pub mod event;
pub mod evidence;
pub mod memory;
pub mod message;
pub mod model;
pub mod policy;
pub mod retrieval;
pub mod sandbox;
pub mod tool;
pub mod trace;

pub use context::RequestContext;
