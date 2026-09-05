//! Starts, stops, and streams the output of units. Host processes run under `setsid` so a stop
//! kills the whole tree (`cargo run` and the binary it spawned). Compose services are driven
//! through `docker compose` and followed with `logs -f`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc::UnboundedSender;

use crate::core::types::{Event, Kind, LogLine, RunnerError, ServiceState, Stream, Unit};

const COMPOSE_FILE: &str = "deploy/compose.yml";

/// Owns the children the console started.
pub struct Runner {
    root: PathBuf,
    tx: UnboundedSender<Event>,
    /// Process-group ids of host processes and tasks, by unit id.
    groups: HashMap<String, u32>,
    /// `docker compose logs -f` children, by service name.
    followers: HashMap<String, Child>,
}

impl Runner {
    /// A runner working from the repo root.
    pub fn new(root: PathBuf, tx: UnboundedSender<Event>) -> Self {
        Self {
            root,
            tx,
            groups: HashMap::new(),
            followers: HashMap::new(),
        }
    }

    /// Starts a unit. Services come up detached and are then followed.
    pub fn start(&mut self, unit: &Unit) -> Result<(), RunnerError> {
        match &unit.kind {
            Kind::Service { service, profile } => {
                let mut cmd = self.compose(profile.as_deref());
                cmd.args(["up", "-d", service]);
                self.spawn_streaming(&unit.id, cmd, true)?;
                self.follow(service)
            }
            Kind::Process | Kind::Task => {
                let mut cmd = Command::new("setsid");
                cmd.arg("just").args(&unit.args);
                self.spawn_streaming(&unit.id, cmd, true)
            }
        }
    }

    /// Stops a unit. Host trees get SIGTERM; services get `compose stop`.
    pub fn stop(&mut self, unit: &Unit) -> Result<(), RunnerError> {
        match &unit.kind {
            Kind::Service { service, profile } => {
                self.unfollow(service);
                let mut cmd = self.compose(profile.as_deref());
                cmd.args(["stop", service]);
                self.spawn_streaming(&unit.id, cmd, false)
            }
            Kind::Process | Kind::Task => {
                if let Some(pgid) = self.groups.remove(&unit.id) {
                    self.note(&unit.id, format!("stopping process group {pgid}"));
                    let mut kill = Command::new("kill");
                    kill.args(["-TERM", "--", &format!("-{pgid}")]);
                    kill.stdout(Stdio::null()).stderr(Stdio::null());
                    kill.spawn().map_err(|source| RunnerError::Spawn {
                        cmd: format!("kill -TERM -- -{pgid}"),
                        source,
                    })?;
                }
                Ok(())
            }
        }
    }

    /// Whether a host process or task started here is still tracked.
    pub fn owns(&self, unit_id: &str) -> bool {
        self.groups.contains_key(unit_id)
    }

    /// Drops the record of a process group after its child exited.
    pub fn forget(&mut self, unit_id: &str) {
        self.groups.remove(unit_id);
    }

    /// Begins streaming a service's container logs, if not already.
    pub fn follow(&mut self, service: &str) -> Result<(), RunnerError> {
        if self.followers.contains_key(service) {
            return Ok(());
        }
        let mut cmd = self.compose_all_profiles();
        cmd.args(["logs", "-f", "--tail", "200", "--no-color", service]);
        let child = self.spawn_piped(service, cmd, false)?;
        self.followers.insert(service.to_owned(), child);
        Ok(())
    }

    /// Stops following a service's logs.
    pub fn unfollow(&mut self, service: &str) {
        if let Some(mut child) = self.followers.remove(service) {
            let _ = child.start_kill();
        }
    }

    /// Kills every host process and follower. Compose services are left running.
    pub fn shutdown(&mut self) {
        let services: Vec<String> = self.followers.keys().cloned().collect();
        for s in services {
            self.unfollow(&s);
        }
        for pgid in self.groups.values() {
            let _ = std::process::Command::new("kill")
                .args(["-TERM", "--", &format!("-{pgid}")])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        self.groups.clear();
    }

    /// One `docker compose ps -a --format json` snapshot, keyed by service.
    pub async fn service_states(root: &PathBuf) -> Result<HashMap<String, ServiceState>, String> {
        let out = Command::new("docker")
            .args(["compose", "-f", COMPOSE_FILE])
            .args(all_profiles())
            .args(["ps", "-a", "--format", "json"])
            .current_dir(root)
            .output()
            .await
            .map_err(|e| format!("docker compose ps: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "docker compose ps: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        parse_ps(&String::from_utf8_lossy(&out.stdout))
    }

    fn compose(&self, profile: Option<&str>) -> Command {
        let mut cmd = Command::new("docker");
        cmd.args(["compose", "-f", COMPOSE_FILE]);
        if let Some(p) = profile {
            cmd.args(["--profile", p]);
        }
        cmd.current_dir(&self.root);
        cmd
    }

    fn compose_all_profiles(&self) -> Command {
        let mut cmd = Command::new("docker");
        cmd.args(["compose", "-f", COMPOSE_FILE]);
        cmd.args(all_profiles());
        cmd.current_dir(&self.root);
        cmd
    }

    /// Spawns, streams both outputs as log lines, and reports the exit; `track` records the
    /// process group so `stop` can kill it.
    fn spawn_streaming(
        &mut self,
        unit_id: &str,
        cmd: Command,
        track: bool,
    ) -> Result<(), RunnerError> {
        let mut child = self.spawn_piped(unit_id, cmd, true)?;
        if track && let Some(pid) = child.id() {
            self.groups.insert(unit_id.to_owned(), pid);
        }
        let tx = self.tx.clone();
        let id = unit_id.to_owned();
        tokio::spawn(async move {
            let code = match child.wait().await {
                Ok(status) => status.code(),
                Err(e) => {
                    let _ = tx.send(Event::Log {
                        unit: id.clone(),
                        line: LogLine::now(Stream::Meta, format!("wait failed: {e}")),
                    });
                    None
                }
            };
            let _ = tx.send(Event::Exited { unit: id, code });
        });
        Ok(())
    }

    fn spawn_piped(
        &self,
        unit_id: &str,
        mut cmd: Command,
        announce: bool,
    ) -> Result<Child, RunnerError> {
        let line = describe(cmd.as_std());
        cmd.current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(false)
            .env("CARGO_TERM_COLOR", "never")
            .env("NO_COLOR", "1");
        let mut child = cmd.spawn().map_err(|source| RunnerError::Spawn {
            cmd: line.clone(),
            source,
        })?;
        if announce {
            self.note(unit_id, format!("$ {line}"));
        }
        if let Some(out) = child.stdout.take() {
            pump(self.tx.clone(), unit_id.to_owned(), Stream::Out, out);
        }
        if let Some(err) = child.stderr.take() {
            pump(self.tx.clone(), unit_id.to_owned(), Stream::Err, err);
        }
        Ok(child)
    }

    fn note(&self, unit_id: &str, text: String) {
        let _ = self.tx.send(Event::Log {
            unit: unit_id.to_owned(),
            line: LogLine::now(Stream::Meta, text),
        });
    }
}

fn all_profiles() -> [&'static str; 10] {
    [
        "--profile",
        "model",
        "--profile",
        "crawl",
        "--profile",
        "browser",
        "--profile",
        "metrics",
        "--profile",
        "gpu-metrics",
    ]
}

fn describe(cmd: &std::process::Command) -> String {
    let mut parts = vec![cmd.get_program().to_string_lossy().into_owned()];
    parts.extend(cmd.get_args().map(|a| a.to_string_lossy().into_owned()));
    parts.join(" ")
}

fn pump<R>(tx: UnboundedSender<Event>, unit: String, stream: Stream, reader: R)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        loop {
            let text = match lines.next_line().await {
                Ok(Some(text)) => text,
                Ok(None) => break,
                // A pane that quietly stops updating while the process still runs is the worst
                // failure this console can have, so it says why it stopped.
                Err(e) => {
                    let _ = tx.send(Event::Log {
                        unit: unit.clone(),
                        line: LogLine::now(Stream::Meta, format!("log capture ended: {e}")),
                    });
                    break;
                }
            };
            let text = strip_ansi(&text);
            if tx
                .send(Event::Log {
                    unit: unit.clone(),
                    line: LogLine::now(stream, text),
                })
                .is_err()
            {
                break;
            }
        }
    });
}

/// Removes ANSI escape sequences so colored output from children renders as plain text.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for d in chars.by_ref() {
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Parses `docker compose ps --format json`: a JSON array on older releases, one object per
/// line on newer ones. Anything else is an error, not an empty stack.
pub fn parse_ps(raw: &str) -> Result<HashMap<String, ServiceState>, String> {
    #[derive(serde::Deserialize)]
    struct Row {
        #[serde(rename = "Service")]
        service: String,
        #[serde(rename = "State")]
        state: String,
        #[serde(rename = "Health", default)]
        health: String,
        #[serde(rename = "ExitCode", default)]
        exit_code: i32,
    }
    let rows: Vec<Row> = if raw.trim().is_empty() {
        Vec::new()
    } else if let Ok(rows) = serde_json::from_str::<Vec<Row>>(raw) {
        rows
    } else {
        raw.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Row>(l).map_err(|e| format!("compose ps row: {e}")))
            .collect::<Result<_, _>>()?
    };
    Ok(rows
        .into_iter()
        .map(|r| {
            (
                r.service,
                ServiceState {
                    state: r.state,
                    health: r.health,
                    exit_code: r.exit_code,
                },
            )
        })
        .collect())
}
