//! `SparkyAI` Discord bot: slash commands → HTTP calls to the api → streamed replies with citations.
//! A client of the API; never links the harness.
//!
//! Scaffold phase: modules and config fields exist ahead of the code that uses them.
//! Remove this allow when Phase 1 lands.
#![allow(dead_code)]

mod api_client;
mod bot;
mod commands;
mod config;
mod reply;
mod telemetry;

fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let cfg = config::Config::load()?;
    let _guard = telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level);
    tracing::info!(env = %cfg.app.env, api = %cfg.api.base_url, "discord starting");
    bot::run(cfg)
}
