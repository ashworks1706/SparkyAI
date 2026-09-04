//! Every data type in the engine: values that cross module boundaries, wire shapes, config-like
//! settings, and errors. Objects (state plus the methods that own it) live beside their `impl`.

pub mod agent;
pub mod assemble;
pub mod chat;
pub mod context;
pub mod event;
pub mod evidence;
pub mod memory;
pub mod message;
pub mod model;
pub mod policy;
pub mod retrieval;
pub mod store;
pub mod tool;
pub mod trace;
pub mod wire;
