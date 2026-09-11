//! The interfaces every adapter implements. Types are in core::types; implementations live
//! in agent::harness, agent::model, agent::tools, and stores.

pub mod compaction;
pub mod confirmation;
pub mod conversation;
pub mod detector;
pub mod guardrail;
pub mod memory;
pub mod model;
pub mod policy;
pub mod profile;
pub mod query;
pub mod retrieval;
pub mod sandbox;
pub mod skills;
pub mod tool;
pub mod trace;
