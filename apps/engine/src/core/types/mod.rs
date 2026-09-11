//! Every data type in the engine: values that cross module boundaries, wire shapes, config-like
//! settings, and errors. Objects (state plus the methods that own it) live beside their impl.

pub mod agent;
pub mod conversation;
pub mod http;
pub mod knowledge;
pub mod memory;
pub mod model;
pub mod safety;
pub mod store;
pub mod tools;
pub mod trace;
