//! The thing that thinks: harness (types, traits, loop), model adapters, built-in tools.
//! `harness` imports nothing else in the crate; `model` and `tools` import only `harness`.

pub mod harness;
pub mod model;
pub mod tools;
