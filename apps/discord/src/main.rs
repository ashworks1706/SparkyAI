//! SparkyAI Discord bot: slash commands to HTTP calls to the engine to replies with citations.
//! A client of the engine. Never links it.

mod access;
mod analytics;
mod bot;
mod core;
mod engine;
mod render;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if let Err(e) = dotenvy::dotenv()
        && !e.not_found()
    {
        return Err(anyhow::anyhow!(".env: {e}"));
    }
    let cfg = core::config::Config::load()?;
    let _guard = core::telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level)?;
    tracing::info!(env = %cfg.app.env, engine = %cfg.engine.base_url, "discord starting");
    bot::run(cfg).await
}
