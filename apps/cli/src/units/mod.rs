//! What the console runs: the unit catalog, process runner, output parsing, logs, probes, sandbox.

pub mod catalog;
pub mod health;
pub mod logs;
pub mod output;
pub mod runner;
pub mod sandbox;

pub use catalog::{catalog, sandbox_session, sandbox_switch, task};
