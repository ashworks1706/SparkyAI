//! The interfaces every adapter implements. Types are in `core::types`; implementations live
//! in `agent::harness`, `agent::model`, `agent::tools`, and `stores`.

pub mod confirmation;
pub mod conversation;
pub mod memory;
pub mod model;
pub mod policy;
pub mod retrieval;
pub mod tool;
pub mod trace;
