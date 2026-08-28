//! `sparky-api`: axum HTTP API over the harness. Runs migrations on startup (Phase 2), then serves.

mod admin;
mod health;
mod routes;
mod wiring;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (cfg, _guard) = sparky_runtime::bootstrap()?;
    tracing::info!(env = %cfg.app.env, "sparky-api starting");
    wiring::serve(cfg).await
}
