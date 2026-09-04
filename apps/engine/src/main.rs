//! `SparkyAI` engine. One binary: the agent, its HTTP surface, and the Postgres adapters
//! behind the harness store traits.
//! Module boundaries (see `docs/ARCHITECTURE.md`): `core` imports nothing else in this crate;
//! `agent::harness` imports only `core`; `agent::{model,tools}` and `stores` import `core` and
//! `agent::harness`; `routes` and `wiring` compose them.
//!
//! Scaffold phase: modules and config fields exist ahead of the code that uses them.
//! Remove this allow when Phase 1 lands.
#![allow(dead_code)]

mod agent;
mod core;
mod routes;
mod stores;
mod wiring;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let cfg = core::config::Config::load()?;
    let _guard = core::telemetry::init(&cfg.telemetry, "engine", &cfg.app.env, &cfg.app.log_level)?;
    tracing::info!(env = %cfg.app.env, "engine starting");
    wiring::serve(cfg).await
}
