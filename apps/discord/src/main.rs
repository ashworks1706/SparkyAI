//! `SparkyAI` Discord bot: slash commands → HTTP calls to the engine → streamed replies with citations.
//! A client of the engine; never links it.
//!
//! Scaffold phase: modules and config fields exist ahead of the code that uses them.
//! Remove this allow when Phase 1 lands.
#![allow(dead_code)]

mod bot;
mod commands;
mod config;
mod engine_client;
mod reply;
mod telemetry;

fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let cfg = config::Config::load()?;
    let _guard = telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level);
    tracing::info!(env = %cfg.app.env, engine = %cfg.engine.base_url, "discord starting");
    bot::run(cfg)
}
