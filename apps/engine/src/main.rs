//! SparkyAI engine: agent, HTTP surface, and store adapters behind the harness traits.
#![cfg_attr(not(feature = "standalone"), allow(dead_code))]

mod core;
mod routes;
mod runtime;
mod stores;
mod wiring;

/// Starts telemetry outside the tokio runtime, serves, then drops the runtime before telemetry.
fn main() -> anyhow::Result<()> {
    if let Err(e) = dotenvy::dotenv()
        && !e.not_found()
    {
        return Err(anyhow::anyhow!(".env: {e}"));
    }
    let cfg = core::config::Config::load()?;
    let guard = core::telemetry::init(&cfg.telemetry, "engine", &cfg.app.env, &cfg.app.log_level)?;
    tracing::info!(env = %cfg.app.env, "engine starting");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let served = runtime.block_on(wiring::serve(cfg));
    drop(runtime);
    drop(guard);
    served
}
