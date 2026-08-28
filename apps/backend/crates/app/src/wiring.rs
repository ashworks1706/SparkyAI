//! Construct concrete adapters and hand them to the harness. One function per process.

use crate::config::Config;

/// HTTP API process: harness + routes. Runs until shutdown.
pub async fn api(cfg: Config) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(&cfg.app.http_addr).await?;
    tracing::info!(addr = %cfg.app.http_addr, "api listening");
    axum::serve(listener, sparky_server::routes::router()).await?;
    Ok(())
}

/// Discord bot process: serenity client that calls the API over HTTP.
pub fn discord(_cfg: Config) -> anyhow::Result<()> {
    anyhow::bail!("discord bot not implemented")
}

/// Applies pending SQL migrations.
pub fn migrate(_cfg: Config) -> anyhow::Result<()> {
    anyhow::bail!("migrate not implemented")
}

/// Local development: api and discord in one process.
pub async fn dev(cfg: Config) -> anyhow::Result<()> {
    api(cfg).await
}
