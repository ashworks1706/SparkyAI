//! The interfaces every adapter implements. Types are in core::types; implementations live
//! in runtime::harness, runtime::model, runtime::tools, and stores.

pub mod conversation;
pub mod knowledge;
pub mod memory;
pub mod model;
pub mod safety;
pub mod tools;
pub mod trace;
