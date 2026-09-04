//! `SparkyAI` engine. One binary: the agent, its HTTP surface, and the Postgres adapters
//! behind the harness store traits.
//! Module boundaries (see `docs/ARCHITECTURE.md`): `core` imports nothing else in this crate;
//! `agent::harness` imports only `core`; `agent::{model,tools}` and `stores` import `core` and
//! `agent::harness`; `routes` and `wiring` compose them.

mod agent;
mod core;
mod routes;
mod stores;
mod wiring;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if let Err(e) = dotenvy::dotenv()
        && !e.not_found()
    {
        return Err(anyhow::anyhow!(".env: {e}"));
    }
    let cfg = core::config::Config::load()?;
    let _guard = core::telemetry::init(&cfg.telemetry, "engine", &cfg.app.env, &cfg.app.log_level)?;
    tracing::info!(env = %cfg.app.env, "engine starting");
    wiring::serve(cfg).await
}
