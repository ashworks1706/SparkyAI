//! `SparkyAI` backend. One binary: HTTP API over the harness, with retrieval, memory, and storage.
//! Module boundaries (see `docs/ARCHITECTURE.md`): `harness` imports nothing else in this crate;
//! `model`, `tools`, `retrieval`, `storage` import only `harness`; `routes` and `wiring` compose them.
//!
//! Scaffold phase: modules and config fields exist ahead of the code that uses them.
//! Remove this allow when Phase 1 lands.
#![allow(dead_code)]

mod config;
mod harness;
mod model;
mod retrieval;
mod routes;
mod storage;
mod telemetry;
mod tools;
mod wiring;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let cfg = config::Config::load()?;
    let _guard = telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level)?;
    tracing::info!(env = %cfg.app.env, "api starting");
    wiring::serve(cfg).await
}
