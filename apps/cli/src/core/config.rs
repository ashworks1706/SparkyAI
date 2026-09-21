//! Settings from SPARKY_<SECTION>__<KEY> env vars. Every field defaults; engine .env is honored.

use std::path::{Path, PathBuf};

use figment::Figment;
use figment::providers::{Env, Format, Toml};
use secrecy::SecretString;
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
    /// Bearer token the engine wants. The console reads the same .env the engine does.
    pub service_token: SecretString,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".into(),
            service_token: SecretString::from(""),
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
    /// Phoenix trace UI, probed and shown in the status bar.
    pub phoenix_url: String,
    /// Lines kept per unit.
    pub log_lines: usize,
    /// Directory for persistent unit logs.
    pub log_dir: PathBuf,
    /// Size a unit log file reaches before it is rotated to one .1 file, in mebibytes.
    pub log_file_max_mb: u64,
    /// Longest line kept from a unit's output; the rest of a longer line is dropped.
    pub log_line_chars: usize,
    /// Seconds between health probes.
    pub health_interval_secs: u64,
    /// Milliseconds a port check waits before treating the port as free.
    pub port_check_ms: u64,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            phoenix_url: "http://localhost:6006".into(),
            log_lines: 5000,
            log_dir: PathBuf::from(".sparky/logs"),
            log_file_max_mb: 20,
            log_line_chars: 4_000,
            health_interval_secs: 5,
            port_check_ms: 150,
        }
    }
}

/// TOML layer read when SPARKY_CONFIG_FILE is unset. Missing is not an error.
pub const DEFAULT_CONFIG_FILE: &str = "sparky.toml";

/// Loads settings from the TOML layer then the environment, which wins.
pub fn load() -> anyhow::Result<Config> {
    let path = match std::env::var("SPARKY_CONFIG_FILE") {
        Ok(path) => path,
        Err(std::env::VarError::NotPresent) => DEFAULT_CONFIG_FILE.to_owned(),
        Err(e) => return Err(anyhow::anyhow!("SPARKY_CONFIG_FILE: {e}")),
    };
    Figment::new()
        .merge(Toml::file(path))
        .merge(Env::prefixed("SPARKY_").split("__"))
        .extract()
        .map_err(|e| anyhow::anyhow!("config: {e}"))
}

/// Walks up from start to the directory holding the repo justfile.
pub fn repo_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join("justfile").is_file() && dir.join("apps").is_dir())
        .map(Path::to_path_buf)
}
