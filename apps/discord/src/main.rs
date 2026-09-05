//! `SparkyAI` Discord bot: slash commands → HTTP calls to the engine → replies with citations.
//! A client of the engine; never links it.

mod bot;
mod commands;
mod core;
mod engine_client;
mod reply;
mod sse;

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
