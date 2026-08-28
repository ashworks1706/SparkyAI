//! `SparkyAI` engine. One binary: the agent, its knowledge, its storage, and the HTTP surface.
//! Module boundaries (see `docs/ARCHITECTURE.md`): `agent::harness` imports nothing else in this
//! crate; `agent::{model,tools}`, `knowledge`, `storage` import only `agent::harness`; `routes` and
//! `wiring` compose them.
//!
//! Scaffold phase: modules and config fields exist ahead of the code that uses them.
//! Remove this allow when Phase 1 lands.
#![allow(dead_code)]

mod agent;
mod config;
mod knowledge;
mod routes;
mod storage;
mod telemetry;
mod wiring;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let cfg = config::Config::load()?;
    let _guard = telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level)?;
    tracing::info!(env = %cfg.app.env, "engine starting");
    wiring::serve(cfg).await
}
