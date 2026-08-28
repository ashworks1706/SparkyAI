//! `SparkyAI` engine. One binary: the agent and its HTTP surface. Holds no database connections —
//! every store is behind `apps/knowledge`.
//! Module boundaries (see `docs/ARCHITECTURE.md`): `agent::harness` imports nothing else in this
//! crate; `agent::{model,tools}` and `clients` import only `agent::harness`; `routes` and `wiring`
//! compose them.
//!
//! Scaffold phase: modules and config fields exist ahead of the code that uses them.
//! Remove this allow when Phase 1 lands.
#![allow(dead_code)]

mod agent;
mod clients;
mod config;
mod routes;
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
