//! Composition root. Loads config, starts telemetry, wires adapters into the harness, runs Discord + HTTP.

mod config;
mod telemetry;
mod wiring;

use clap::Parser;

#[derive(Parser)]
#[command(name = "sparky", version, about = "SparkyAI backend")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Subcommand)]
enum Cmd {
    /// HTTP API: harness, policy, retrieval, memory. The only process that talks to models and stores.
    Api,
    /// Discord bot; an HTTP client of `api`.
    Discord,
    /// Apply database migrations.
    Migrate,
    /// Run api + discord in one process for local development.
    Dev,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    dotenvy::dotenv().ok();
    let cfg = config::Config::load()?;
    let _guard = telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level)?;
    tracing::info!(env = %cfg.app.env, "starting");

    match cli.cmd {
        Cmd::Api => wiring::api(cfg).await,
        Cmd::Discord => wiring::discord(cfg),
        Cmd::Migrate => wiring::migrate(cfg),
        Cmd::Dev => wiring::dev(cfg).await,
    }
}
