//! Composition root. Loads config, starts telemetry, wires adapters into the harness, runs Discord + HTTP.

mod config;
mod telemetry;
mod wiring;

use clap::Parser;

#[derive(Parser)]
#[command(name = "sparky", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Subcommand)]
enum Cmd {
    /// Run the Discord bot and HTTP server.
    Serve,
    /// Run offline ingestion jobs.
    Ingest,
    /// Apply database migrations.
    Migrate,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    dotenvy::dotenv().ok();
    let cfg = config::Config::load()?;
    let _guard = telemetry::init(&cfg.telemetry, &cfg.app.env, &cfg.app.log_level)?;
    tracing::info!(env = %cfg.app.env, "starting");

    match cli.cmd {
        Cmd::Serve => wiring::serve(cfg).await,
        Cmd::Ingest => wiring::ingest(cfg),
        Cmd::Migrate => wiring::migrate(cfg),
    }
}
