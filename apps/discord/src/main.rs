//! `SparkyAI` Discord bot: slash commands → HTTP calls to the engine → replies with citations.
//! A client of the engine; never links it.

mod bot;
mod commands;
mod core;
mod engine_client;
mod reply;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let cfg = core::config::Config::load()?;
    let _guard = core::telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level)?;
    tracing::info!(env = %cfg.app.env, engine = %cfg.engine.base_url, "discord starting");
    bot::run(cfg).await
}
