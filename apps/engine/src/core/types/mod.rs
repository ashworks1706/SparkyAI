//! Every struct and enum in the engine. Behaviour lives in the modules that use them;
//! nothing outside `core` defines a type.

pub mod adapters;
pub mod agent;
pub mod assemble;
pub mod chat;
pub mod context;
pub mod event;
pub mod evidence;
pub mod harness;
pub mod memory;
pub mod message;
pub mod model;
pub mod policy;
pub mod retrieval;
pub mod sandbox;
pub mod store;
pub mod tool;
pub mod trace;
