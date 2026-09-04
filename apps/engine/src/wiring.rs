//! Construct concrete adapters, hand them to the harness, build the router, serve.

use crate::core::config::Config;

/// Serves until shutdown.
pub async fn serve(cfg: Config) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(&cfg.app.http_addr).await?;
    tracing::info!(addr = %cfg.app.http_addr, "listening");
    axum::serve(listener, crate::routes::router()).await?;
    Ok(())
}
