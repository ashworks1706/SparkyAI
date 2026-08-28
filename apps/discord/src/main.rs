//! `sparky-discord`: slash commands → HTTP calls to `sparky-api` → streamed replies with citations.
//! A client of the API; never links the harness.

mod api_client;
mod bot;
mod commands;
mod reply;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (cfg, _guard) = sparky_runtime::bootstrap()?;
    tracing::info!(env = %cfg.app.env, api = %cfg.api.base_url, "sparky-discord starting");
    bot::run(cfg)
}
