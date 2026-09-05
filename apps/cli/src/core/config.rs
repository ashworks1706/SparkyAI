//! Settings from `SPARKY_<SECTION>__<KEY>` env vars. Every field has a default so the console
//! runs from a fresh clone; the engine's own `.env` is honoured when present.

use std::path::{Path, PathBuf};

use figment::Figment;
use figment::providers::Env;
use serde::Deserialize;

/// Console settings.
#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    /// Engine location and service token.
    pub engine: Engine,
    /// Chat model server, for the health probe.
    pub model: Model,
    /// Console-only knobs.
    pub cli: Cli,
}

/// Engine settings.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Engine {
    /// Base URL of the engine.
    pub base_url: String,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".into(),
        }
    }
}

/// Model server settings.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Model {
    /// OpenAI-compatible base URL of the chat server.
    pub base_url: String,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8000/v1".into(),
        }
    }
}

/// Console settings.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Cli {
    /// Phoenix UI, probed for health and opened with `o`.
    pub phoenix_url: String,
    /// Lines kept per unit.
    pub log_lines: usize,
    /// Directory for persistent unit logs.
    pub log_dir: PathBuf,
    /// Seconds between health probes.
    pub health_interval_secs: u64,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            phoenix_url: "http://localhost:6006".into(),
            log_lines: 5000,
            log_dir: PathBuf::from(".sparky/logs"),
            health_interval_secs: 5,
        }
    }
}

/// Loads settings from the environment.
pub fn load() -> anyhow::Result<Config> {
    Figment::new()
        .merge(Env::prefixed("SPARKY_").split("__"))
        .extract()
        .map_err(|e| anyhow::anyhow!("config: {e}"))
}

/// Walks up from `start` to the directory holding the repo's `justfile`.
pub fn repo_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join("justfile").is_file() && dir.join("apps").is_dir())
        .map(Path::to_path_buf)
}
