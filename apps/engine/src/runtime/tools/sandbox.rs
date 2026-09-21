//! A command in an isolated container: read-only root, capped resources, non-root user, and either
//! no network or an internal network whose only way out is the egress proxy.

use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::core::traits::tools::Tool;
use crate::core::traits::tools::sandbox::Sandbox;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::sandbox::{
    SandboxCommand, SandboxError, SandboxOutput, SandboxRequest, SandboxSession, session_name,
    workspace_path,
};
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::harness::safety::redact::redact_text;
use crate::runtime::tools::structured;

/// Name of the tool that runs a command in the isolated environment.
pub const SANDBOX: &str = "run_sandbox";
/// Directory the workspace is mounted at inside the container.
pub const WORKSPACE: &str = "/tmp";

/// Port the egress proxy listens on.
pub const PROXY_PORT: u16 = 3128;

/// The way out of the sandbox when commands may reach the public internet.
#[derive(Debug, Clone)]
pub struct Egress {
    /// Internal network the containers join. It has no route out of its own.
    pub network: String,
    /// Image of the proxy that joins the network and the outside.
    pub proxy_image: String,
    /// Container name of the proxy, its host name on the network.
    pub proxy_name: String,
}

impl Egress {
    /// The proxy address the containers are handed.
    fn proxy_url(&self) -> String {
        format!("http://{}:{PROXY_PORT}", self.proxy_name)
    }
}

/// How the sandbox is started and what it may consume.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Container runtime binary.
    pub runtime: String,
    /// Image the command runs in.
    pub image: String,
    /// Memory ceiling, in the form the runtime accepts.
    pub memory: String,
    /// CPU ceiling, in the form the runtime accepts.
    pub cpus: String,
    /// Process ceiling.
    pub pids: u32,
    /// Wall-clock budget.
    pub timeout: Duration,
    /// Longest stdout or stderr handed back.
    pub max_output_chars: usize,
    /// How long a session container stays up with nothing running in it.
    pub session_idle_secs: u64,
    /// Sessions one caller may hold at once.
    pub max_sessions: usize,
    /// Session containers across every caller.
    pub max_sessions_total: usize,
    /// Commands running at once across every caller.
    pub max_running: usize,
    /// Label value naming this engine's containers apart from another engine's.
    pub instance: String,
    /// Size of the writable workspace, in mebibytes.
    pub workspace_mb: u32,
    /// Commands kept for the operator view.
    pub recent_commands: usize,
    /// The way out to the public internet. None runs with no network.
    pub egress: Option<Egress>,
}

impl Default for Limits {
    fn default() -> Self {
        Self::from(&crate::core::config::SandboxSettings::default())
    }
}

impl From<&crate::core::config::SandboxSettings> for Limits {
    fn from(cfg: &crate::core::config::SandboxSettings) -> Self {
        Self {
            runtime: cfg.runtime.clone(),
            image: cfg.image.clone(),
            memory: cfg.memory.clone(),
            cpus: cfg.cpus.clone(),
            pids: cfg.pids,
            timeout: Duration::from_secs(cfg.timeout_secs),
            max_output_chars: cfg.max_output_chars,
            session_idle_secs: cfg.session_idle_secs,
            max_sessions: cfg.max_sessions,
            max_sessions_total: cfg.max_sessions_total,
            max_running: cfg.max_running,
            instance: cfg.instance.clone(),
            workspace_mb: cfg.workspace_mb,
            recent_commands: cfg.recent_commands,
            egress: cfg.egress.then(|| Egress {
                network: cfg.egress_network.clone(),
                proxy_image: cfg.egress_proxy_image.clone(),
                proxy_name: cfg.egress_proxy_name.clone(),
            }),
        }
    }
}

/// The guard of a lock, taken back after a panic. What it holds is bookkeeping, not an
/// invariant another thread could have half written.
fn held<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    lock.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Label every container the engine starts carries, so an operator can find them all.
pub const LABEL: &str = "sparky.sandbox";

/// Label naming which engine instance started a container.
pub const INSTANCE_LABEL: &str = "sparky.sandbox.instance";

/// Prefix of every session container name.
const SESSION_PREFIX: &str = "sparky-sb-";

/// What the engine knows about one live session container.
#[derive(Debug, Clone)]
struct Live {
    /// Name the caller gave it.
    session: String,
    /// When it was started here.
    started: Instant,
    /// When a command last ran in it.
    used: Instant,
    /// Commands run in it.
    runs: u64,
}

impl Live {
    fn new(session: &str) -> Self {
        let now = Instant::now();
        Self {
            session: session.to_owned(),
            started: now,
            used: now,
            runs: 0,
        }
    }
}

/// Holds one command in the running list for as long as the call that started it lives.
///
/// The loop drops a tool call that is cancelled, so the command it was running has to come off
/// the list on the way out rather than at the end of a body that never runs.
struct Running {
    sandbox: ContainerSandbox,
    ticket: u64,
    started: Instant,
}

impl Running {
    /// Moves the finished command into the log and ends the guard.
    fn ended(self, exit_code: Option<i32>, duration_ms: u64) {
        let Some(mut command) = self.sandbox.take_running(self.ticket) else {
            return;
        };
        command.exit_code = exit_code;
        command.duration_ms = Some(duration_ms);
        self.sandbox.keep(command);
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        // A cancelled call ends here. It is recorded as ended with no exit status.
        if let Some(mut command) = self.sandbox.take_running(self.ticket) {
            command.duration_ms = Some(
                self.started
                    .elapsed()
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX),
            );
            self.sandbox.keep(command);
        }
    }
}

/// Runs commands in a container. Clones share the session registry and the command log.
#[derive(Debug, Clone)]
pub struct ContainerSandbox {
    limits: Limits,
    /// Container name to what is known about it, for the idle sweep and the per-caller cap.
    sessions: Arc<Mutex<HashMap<String, Live>>>,
    /// The commands that ran, newest last, capped at limits.recent_commands.
    recent: Arc<Mutex<VecDeque<SandboxCommand>>>,
    /// Commands running right now, by the ticket they took.
    running: Arc<Mutex<Vec<(u64, SandboxCommand)>>>,
    /// Hands out a ticket per command, so one can be found again when it ends.
    ticket: Arc<AtomicU64>,
    /// Whether the tool is offered. An operator turns it off without restarting the engine.
    enabled: Arc<AtomicBool>,
    /// One permit per command allowed to run at once, across every caller.
    slots: Arc<tokio::sync::Semaphore>,
}

impl Default for ContainerSandbox {
    fn default() -> Self {
        Self::new(Limits::default())
    }
}

impl ContainerSandbox {
    /// Builds the sandbox over its limits.
    pub fn new(limits: Limits) -> Self {
        Self {
            slots: Arc::new(tokio::sync::Semaphore::new(limits.max_running.max(1))),
            limits,
            sessions: Arc::default(),
            recent: Arc::default(),
            running: Arc::default(),
            ticket: Arc::default(),
            enabled: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Records a command as started. The guard takes it out again however the call ends.
    fn command_started(&self, mut command: SandboxCommand) -> Running {
        let ticket = self.ticket.fetch_add(1, Ordering::Relaxed);
        command.id = ticket;
        held(&self.running).push((ticket, command));
        Running {
            sandbox: self.clone(),
            ticket,
            started: Instant::now(),
        }
    }

    /// Takes the command of ticket out of the running list, if it is still there.
    fn take_running(&self, ticket: u64) -> Option<SandboxCommand> {
        let mut running = held(&self.running);
        let at = running.iter().position(|(t, _)| *t == ticket)?;
        Some(running.remove(at).1)
    }

    /// Puts an ended command in the log, dropping the oldest once the log is full.
    fn keep(&self, command: SandboxCommand) {
        let mut recent = held(&self.recent);
        while recent.len() >= self.limits.recent_commands {
            recent.pop_front();
        }
        recent.push_back(command);
    }

    /// The prefix every session container of one caller shares.
    fn owner_prefix(ctx: &RequestContext) -> String {
        let mut hasher = DefaultHasher::new();
        (&ctx.tenant_id, &ctx.user_id).hash(&mut hasher);
        format!("{SESSION_PREFIX}{:016x}-", hasher.finish())
    }

    /// The container name a session runs under, scoped to the tenant and user.
    pub fn container_name(ctx: &RequestContext, session: &str) -> String {
        format!("{}{session}", Self::owner_prefix(ctx))
    }

    /// Whether the runtime answers, with egress made ready. Called once at boot so a broken
    /// sandbox is not offered.
    pub async fn probe(&self) -> Result<(), SandboxError> {
        let out = Command::new(&self.limits.runtime)
            .arg("version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|e| {
                SandboxError::Runtime(format!("{} did not start: {e}", self.limits.runtime))
            })?;
        if out.status.success() {
            return self.prepare_egress().await;
        }
        Err(SandboxError::Runtime(format!(
            "{} answered: {}",
            self.limits.runtime,
            String::from_utf8_lossy(&out.stderr).trim()
        )))
    }

    /// Runs one runtime subcommand and reports whether it succeeded.
    async fn runtime_ok(&self, args: &[&str]) -> bool {
        Command::new(&self.limits.runtime)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .status()
            .await
            .is_ok_and(|status| status.success())
    }

    /// Whether a container of this name exists and is running.
    async fn running(&self, name: &str) -> bool {
        let out = Command::new(&self.limits.runtime)
            .args(["inspect", "--format", "{{.State.Running}}", name])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .output()
            .await;
        out.is_ok_and(|out| {
            out.status.success() && String::from_utf8_lossy(&out.stdout).trim() == "true"
        })
    }

    /// Removes a container and forgets it.
    async fn remove(&self, name: &str) {
        self.runtime_ok(&["rm", "--force", name]).await;
        held(&self.sessions).remove(name);
    }

    /// Records a session as used now, starting its entry when this is the first time.
    pub(crate) fn note_used(&self, name: &str, session: &str, ran: bool) {
        let mut sessions = held(&self.sessions);
        let live = sessions
            .entry(name.to_owned())
            .or_insert_with(|| Live::new(session));
        live.used = Instant::now();
        if ran {
            live.runs += 1;
        }
    }

    /// The caller's least recently used session, when they already hold the most they may.
    pub(crate) fn least_recently_used(&self, ctx: &RequestContext) -> Option<String> {
        let prefix = Self::owner_prefix(ctx);
        let sessions = held(&self.sessions);
        let mut candidates: Vec<(&String, &Live)> = sessions
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .collect();
        if candidates.len() < self.limits.max_sessions {
            return None;
        }
        candidates.sort_by_key(|(_, live)| live.used);
        candidates.first().map(|(name, _)| (*name).clone())
    }

    /// The least recently used session of every caller, when the engine holds the most it may.
    pub(crate) fn least_recently_used_overall(&self) -> Option<String> {
        let sessions = held(&self.sessions);
        if sessions.len() < self.limits.max_sessions_total.max(1) {
            return None;
        }
        sessions
            .iter()
            .min_by_key(|(_, live)| live.used)
            .map(|(name, _)| name.clone())
    }

    /// Removes every session container of this instance the engine is not tracking: those a
    /// previous process of it started and nothing will reap. Called at boot and on every sweep.
    pub async fn remove_orphans(&self) {
        let filter = format!("label={INSTANCE_LABEL}={}", self.limits.instance);
        let Some(listed) = self
            .inspect(&["ps", "--all", "--filter", &filter, "--format", "{{.Names}}"])
            .await
        else {
            return;
        };
        let orphans: Vec<String> = {
            let sessions = held(&self.sessions);
            listed
                .lines()
                .map(str::trim)
                .filter(|name| name.starts_with(SESSION_PREFIX) && !sessions.contains_key(*name))
                .map(str::to_owned)
                .collect()
        };
        for name in orphans {
            tracing::info!(session = %name, "orphaned sandbox session removed");
            self.runtime_ok(&["rm", "--force", &name]).await;
        }
    }

    /// The sessions idle past their budget.
    pub(crate) fn idle_sessions(&self) -> Vec<String> {
        let budget = Duration::from_secs(self.limits.session_idle_secs);
        held(&self.sessions)
            .iter()
            .filter(|(_, live)| live.used.elapsed() >= budget)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Removes every session idle past its budget.
    pub async fn reap_idle(&self) {
        for name in self.idle_sessions() {
            tracing::debug!(session = %name, "sandbox session reaped");
            self.remove(&name).await;
        }
    }

    /// Starts a session container, or resumes the one already running under this name.
    async fn start_session(
        &self,
        ctx: &RequestContext,
        name: &str,
        session: &str,
    ) -> Result<(), SandboxError> {
        if self.running(name).await {
            self.note_used(name, session, false);
            return Ok(());
        }
        // A container under this name exists but has stopped; its workspace is already gone.
        self.remove(name).await;
        if let Some(lru) = self.least_recently_used(ctx) {
            self.remove(&lru).await;
        }
        if let Some(lru) = self.least_recently_used_overall() {
            tracing::info!(session = %lru, "sandbox session cap reached; least recently used removed");
            self.remove(&lru).await;
        }
        // Registered before it starts, so the orphan sweep never takes a container mid-start.
        self.note_used(name, session, false);
        let out = Command::new(&self.limits.runtime)
            .arg("run")
            .arg("--detach")
            .arg("--name")
            .arg(name)
            .args(self.seal())
            .arg(&self.limits.image)
            .args(["sleep", "infinity"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|e| {
                SandboxError::Runtime(format!("{} did not start: {e}", self.limits.runtime))
            })?;
        let why = String::from_utf8_lossy(&out.stderr);
        // Another request started the same session between the check and here.
        if out.status.success() || why.contains("already in use") {
            self.note_used(name, session, false);
            return Ok(());
        }
        held(&self.sessions).remove(name);
        Err(SandboxError::Runtime(format!(
            "could not start the session: {}",
            why.trim()
        )))
    }

    /// The arguments that seal a container.
    pub fn seal(&self) -> Vec<String> {
        let l = &self.limits;
        let network = l.egress.as_ref().map_or("none", |e| e.network.as_str());
        let mut args: Vec<String> = [
            "--label",
            LABEL,
            "--network",
            network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--memory",
            &l.memory,
            "--cpus",
            &l.cpus,
            "--pids-limit",
            &l.pids.to_string(),
            "--tmpfs",
            &format!("{WORKSPACE}:rw,noexec,nosuid,size={}m", l.workspace_mb),
            "--workdir",
            WORKSPACE,
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        args.push("--label".to_owned());
        args.push(format!("{INSTANCE_LABEL}={}", l.instance));
        if let Some(egress) = &l.egress {
            let url = egress.proxy_url();
            for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"] {
                args.push("--env".to_owned());
                args.push(format!("{var}={url}"));
            }
        }
        args
    }

    /// Makes the egress network and its proxy ready, when egress is on.
    ///
    /// The network is created internal if missing, and refused if it exists and is not internal:
    /// a network with its own route out would bypass the proxy.
    pub async fn prepare_egress(&self) -> Result<(), SandboxError> {
        let Some(egress) = &self.limits.egress else {
            return Ok(());
        };
        let internal = self
            .inspect(&[
                "network",
                "inspect",
                "--format",
                "{{.Internal}}",
                &egress.network,
            ])
            .await;
        match internal.as_deref() {
            Some("true") => {}
            Some(other) => {
                return Err(SandboxError::Runtime(format!(
                    "network {} is not internal (Internal={other}); remove it or name another \
                     in sandbox.egress_network",
                    egress.network
                )));
            }
            None => {
                self.runtime(&["network", "create", "--internal", &egress.network])
                    .await?;
            }
        }
        if !self.running(&egress.proxy_name).await {
            self.runtime_ok(&["rm", "--force", &egress.proxy_name])
                .await;
            self.runtime(&[
                "run",
                "--detach",
                "--label",
                LABEL,
                "--name",
                &egress.proxy_name,
                "--restart",
                "unless-stopped",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "128m",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/var/run/squid",
                "--tmpfs",
                "/var/log/squid",
                "--tmpfs",
                "/var/spool/squid",
                &egress.proxy_image,
            ])
            .await?;
        }
        let joined = self
            .inspect(&[
                "inspect",
                "--format",
                &format!(
                    "{{{{index .NetworkSettings.Networks {:?}}}}}",
                    egress.network
                ),
                &egress.proxy_name,
            ])
            .await;
        if joined.is_none_or(|j| j == "<nil>" || j.is_empty()) {
            self.runtime(&["network", "connect", &egress.network, &egress.proxy_name])
                .await?;
        }
        Ok(())
    }

    /// Runs one runtime subcommand, failing with what it printed.
    async fn runtime(&self, args: &[&str]) -> Result<(), SandboxError> {
        let out = Command::new(&self.limits.runtime)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|e| {
                SandboxError::Runtime(format!("{} did not start: {e}", self.limits.runtime))
            })?;
        if out.status.success() {
            return Ok(());
        }
        Err(SandboxError::Runtime(format!(
            "{} {} failed: {}",
            self.limits.runtime,
            args.first().copied().unwrap_or_default(),
            String::from_utf8_lossy(&out.stderr).trim()
        )))
    }

    /// The trimmed output of a runtime subcommand, or None when it failed.
    async fn inspect(&self, args: &[&str]) -> Option<String> {
        let out = Command::new(&self.limits.runtime)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .output()
            .await
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    }

    /// The arguments for a call that keeps no session.
    pub fn args(&self) -> Vec<String> {
        let mut args = vec!["run".to_owned(), "--rm".to_owned()];
        args.extend(self.seal());
        args.push(self.limits.image.clone());
        args
    }
}

/// Keeps the head and the tail of text, marking what was dropped between them.
///
/// A long run says most in its first lines and its last; keeping only the head loses the exit.
pub(crate) fn clip(text: &str, max: usize) -> String {
    if max == 0 || text.chars().count() <= max {
        return text.to_owned();
    }
    let head_len = max.div_ceil(2);
    let tail_len = max - head_len;
    let chars: Vec<char> = text.chars().collect();
    let dropped = chars.len() - max;
    let head: String = chars.iter().take(head_len).collect();
    let tail: String = chars.iter().skip(chars.len() - tail_len).collect();
    format!("{head}\n[{dropped} characters cut]\n{tail}")
}

/// What a command produced, each stream held to its head and tail of cap bytes.
struct Bounded {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Runs command and reads both streams as they arrive, keeping at most cap bytes of each
/// stream's head and cap of its tail, so a command that prints without end costs bounded memory.
async fn bounded_output(mut command: Command, cap: usize) -> std::io::Result<Bounded> {
    let mut child = command.spawn()?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (out, err, status) = tokio::join!(
        read_bounded(stdout, cap),
        read_bounded(stderr, cap),
        child.wait()
    );
    Ok(Bounded {
        status: status?,
        stdout: out,
        stderr: err,
    })
}

/// The head and tail of a stream, cap bytes each, joined by a line naming what was dropped.
async fn read_bounded<R: tokio::io::AsyncRead + Unpin>(stream: Option<R>, cap: usize) -> Vec<u8> {
    use tokio::io::AsyncReadExt;

    let Some(mut stream) = stream else {
        return Vec::new();
    };
    let mut head: Vec<u8> = Vec::new();
    let mut tail: VecDeque<u8> = VecDeque::new();
    let mut dropped: usize = 0;
    let mut buf = vec![0u8; 8192];
    loop {
        let n = match stream.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let chunk = buf.get(..n).unwrap_or_default();
        let room = cap.saturating_sub(head.len()).min(chunk.len());
        let (first, rest) = chunk.split_at(room);
        head.extend_from_slice(first);
        tail.extend(rest);
        if tail.len() > cap {
            let excess = tail.len() - cap;
            tail.drain(..excess);
            dropped += excess;
        }
    }
    if dropped > 0 {
        head.extend_from_slice(format!("\n[{dropped} bytes cut]\n").as_bytes());
    }
    head.extend(tail);
    head
}

/// The shell that writes standard input to path, with no interpolation of the content.
fn write_to(path: &str) -> String {
    format!("cat > {path}")
}

#[async_trait]
impl Sandbox for ContainerSandbox {
    async fn run(
        &self,
        ctx: &RequestContext,
        request: &SandboxRequest,
    ) -> Result<SandboxOutput, SandboxError> {
        if request.command.trim().is_empty() {
            return Err(SandboxError::Refused("the command is empty".into()));
        }
        let named = request.session.as_deref().map(session_name).transpose()?;
        let session = named.as_deref().map(|name| Self::container_name(ctx, name));
        let mut command = Command::new(&self.limits.runtime);
        match (&session, &named) {
            (Some(container), Some(name)) => {
                self.start_session(ctx, container, name).await?;
                self.note_used(container, name, true);
                command
                    .arg("exec")
                    .arg("--workdir")
                    .arg(WORKSPACE)
                    .arg(container);
            }
            _ => {
                command.args(self.args());
            }
        }
        let budget = self.limits.timeout.min(ctx.remaining());
        // The runtime client is what a timeout kills here; the container is not. timeout inside
        // it kills the command itself, so nothing outlives its budget in a session.
        command
            .arg("timeout")
            .arg("-s")
            .arg("KILL")
            .arg(budget.as_secs().max(1).to_string())
            .arg("sh")
            .arg("-c")
            .arg(&request.command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let _slot = tokio::time::timeout(budget, self.slots.clone().acquire_owned())
            .await
            .map_err(|_| SandboxError::Timeout)?
            .map_err(|_| SandboxError::Runtime("the sandbox is shutting down".into()))?;
        let started = Instant::now();
        let running = self.command_started(SandboxCommand {
            // command_started stamps the ticket over this.
            id: 0,
            at: Utc::now(),
            container: session.clone(),
            session: named.clone(),
            command: redact_text(&request.command),
            exit_code: None,
            duration_ms: None,
        });
        let cap = self.limits.max_output_chars.saturating_mul(4).max(1024);
        let ran = tokio::time::timeout(budget, bounded_output(command, cap)).await;
        let took = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let output = match ran {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                running.ended(None, took);
                return Err(SandboxError::Runtime(format!(
                    "{} did not start: {e}",
                    self.limits.runtime
                )));
            }
            Err(_) => {
                running.ended(None, took);
                return Err(SandboxError::Timeout);
            }
        };
        running.ended(Some(output.status.code().unwrap_or(-1)), took);

        let max = self.limits.max_output_chars;
        Ok(SandboxOutput {
            // A process ended by a signal has no exit code and reports -1.
            exit_code: output.status.code().unwrap_or(-1),
            stdout: clip(&String::from_utf8_lossy(&output.stdout), max),
            stderr: clip(&String::from_utf8_lossy(&output.stderr), max),
            session: request.session.clone(),
        })
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    fn sessions(&self) -> Vec<SandboxSession> {
        let mut out: Vec<SandboxSession> = held(&self.sessions)
            .iter()
            .map(|(name, live)| SandboxSession {
                name: name.clone(),
                session: live.session.clone(),
                age_secs: live.started.elapsed().as_secs(),
                idle_secs: live.used.elapsed().as_secs(),
                runs: live.runs,
            })
            .collect();
        out.sort_by_key(|s| s.idle_secs);
        out
    }

    fn commands(&self) -> Vec<SandboxCommand> {
        let mut out: Vec<SandboxCommand> =
            held(&self.running).iter().map(|(_, c)| c.clone()).collect();
        out.extend(held(&self.recent).iter().rev().cloned());
        out.sort_by_key(|c| std::cmp::Reverse(c.id));
        out
    }

    async fn kill(&self, name: &str) -> Result<(), SandboxError> {
        if !held(&self.sessions).contains_key(name) {
            return Err(SandboxError::Refused(format!(
                "no session container named {name}"
            )));
        }
        self.remove(name).await;
        Ok(())
    }

    async fn put(
        &self,
        ctx: &RequestContext,
        session: &str,
        name: &str,
        content: &[u8],
    ) -> Result<String, SandboxError> {
        let session = session_name(session)?;
        let file = workspace_path(name)?;
        let container = Self::container_name(ctx, &session);
        self.start_session(ctx, &container, &session).await?;
        let path = format!("{WORKSPACE}/{file}");

        let mut child = Command::new(&self.limits.runtime)
            .args(["exec", "--interactive", "--workdir", WORKSPACE, &container])
            .args(["sh", "-c", &write_to(&path)])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                SandboxError::Runtime(format!("{} did not start: {e}", self.limits.runtime))
            })?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(content)
                .await
                .map_err(|e| SandboxError::Runtime(format!("could not write {path}: {e}")))?;
            stdin
                .shutdown()
                .await
                .map_err(|e| SandboxError::Runtime(format!("could not close {path}: {e}")))?;
        }
        let budget = self.limits.timeout.min(ctx.remaining());
        let out = tokio::time::timeout(budget, child.wait_with_output())
            .await
            .map_err(|_| SandboxError::Timeout)?
            .map_err(|e| SandboxError::Runtime(format!("could not write {path}: {e}")))?;
        if !out.status.success() {
            return Err(SandboxError::Runtime(format!(
                "could not write {path}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(path)
    }
}

/// Removes sessions idle past their budget, forever. Spawned once at wiring.
pub async fn reap_sessions(sandbox: ContainerSandbox, every: Duration) {
    let every = every.max(Duration::from_secs(1));
    loop {
        tokio::time::sleep(every).await;
        sandbox.reap_idle().await;
        sandbox.remove_orphans().await;
    }
}

/// What the tool tells the model about itself.
#[derive(Debug, Clone)]
pub struct Wording {
    /// What the tool is for.
    pub description: String,
    /// What the command argument is.
    pub command: String,
    /// What the session argument is.
    pub session: String,
}

impl From<&crate::core::config::SandboxSettings> for Wording {
    fn from(cfg: &crate::core::config::SandboxSettings) -> Self {
        Self {
            description: cfg.description.clone(),
            command: cfg.command_description.clone(),
            session: cfg.session_description.clone(),
        }
    }
}

/// Offers the sandbox to the model.
pub struct SandboxTool {
    sandbox: Arc<dyn Sandbox>,
    risk: RiskClass,
    wording: Wording,
}

impl SandboxTool {
    /// Builds the tool. The risk class it declares is what Policy gates it by.
    pub fn new(sandbox: Arc<dyn Sandbox>, risk: RiskClass, wording: Wording) -> Self {
        Self {
            sandbox,
            risk,
            wording,
        }
    }
}

#[async_trait]
impl Tool for SandboxTool {
    fn available(&self) -> bool {
        self.sandbox.enabled()
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: SANDBOX.to_owned(),
            description: self.wording.description.clone(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": self.wording.command },
                    "session": { "type": "string", "description": self.wording.session }
                },
                "required": ["command"]
            }),
            risk: self.risk,
            // Sandbox calls within one step run one at a time.
            sequential: true,
            timeout_secs: None,
        }
    }

    async fn call(&self, ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        let request: SandboxRequest =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        let out = self.sandbox.run(ctx, &request).await.map_err(|e| match e {
            SandboxError::Refused(reason) => ToolError::InvalidArguments(reason),
            SandboxError::Timeout => ToolError::Timeout,
            run @ SandboxError::Runtime(_) => ToolError::Failed(run.to_string()),
        })?;
        let mut text = format!("exit {}", out.exit_code);
        if !out.stdout.trim().is_empty() {
            text.push_str("\nstdout:\n");
            text.push_str(&out.stdout);
        }
        if !out.stderr.trim().is_empty() {
            text.push_str("\nstderr:\n");
            text.push_str(&out.stderr);
        }
        Ok(ToolOutput {
            content: text,
            data: structured(&out),
            sources: Vec::new(),
        })
    }
}
