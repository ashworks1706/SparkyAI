//! The engine test suite. One file per unit under test; shared doubles in support.

mod agent;
#[cfg(feature = "standalone")]
mod config;
mod conversation;
mod http;
mod knowledge;
mod memory;
mod model;
mod platform;
#[cfg(feature = "standalone")]
mod postgres;
mod safety;
mod support;
mod tools;
mod trace;
