//! Construct concrete adapters and hand them to the harness.

use crate::config::Config;

/// Runs the Discord bot and HTTP server until shutdown.
pub async fn serve(cfg: Config) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(&cfg.app.http_addr).await?;
    tracing::info!(addr = %cfg.app.http_addr, "http listening");
    axum::serve(listener, sparky_server::routes::router()).await?;
    Ok(())
}

/// Runs offline ingestion jobs once and exits.
pub fn ingest(_cfg: Config) -> anyhow::Result<()> {
    anyhow::bail!("ingestion not implemented")
}

/// Applies pending SQL migrations.
pub fn migrate(_cfg: Config) -> anyhow::Result<()> {
    anyhow::bail!("migrate not implemented")
}
