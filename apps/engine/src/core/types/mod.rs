//! Owned domain types. Every struct and enum the engine passes between modules lives here;
//! provider and wire formats stay private inside the adapters that speak them.

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
pub mod sandbox;
pub mod store;
pub mod tool;
pub mod trace;
