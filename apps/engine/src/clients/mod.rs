//! HTTP clients for the other apps. The engine holds no database connections; every store
//! is behind `apps/knowledge`. Implements the `agent::harness` store traits over HTTP.

pub mod knowledge;
