//! The agent runtime: the harness loop, the model adapters, and the built-in tools.
//! harness, model, and tools each import only core, never each other.

pub mod harness;
pub mod model;
pub mod tools;
