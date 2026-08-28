//! Process bootstrap shared by `apps/api` and `apps/discord`: config loading and telemetry.
//! Not part of the harness; contains nothing request-specific.

pub mod config;
pub mod telemetry;

pub use config::Config;

/// Loads `.env`, parses config, and starts telemetry. Call first in every `main`.
pub fn bootstrap() -> anyhow::Result<(Config, telemetry::Guard)> {
    dotenvy::dotenv().ok();
    let cfg = Config::load()?;
    let guard = telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level)?;
    Ok((cfg, guard))
}
