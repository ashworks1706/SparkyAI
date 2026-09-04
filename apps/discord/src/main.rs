//! `SparkyAI` Discord bot: slash commands → HTTP calls to the engine → replies with citations.
//! A client of the engine; never links it.

mod bot;
mod commands;
mod config;
mod engine_client;
mod reply;
mod telemetry;

#[cfg(test)]
mod tests;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let cfg = config::Config::load()?;
    let _guard = telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level);
    tracing::info!(env = %cfg.app.env, engine = %cfg.engine.base_url, "discord starting");
    bot::run(cfg).await
}
