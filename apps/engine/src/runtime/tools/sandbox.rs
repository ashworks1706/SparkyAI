//! A command in an isolated container: read-only root, capped resources, non-root user, and either
//! no network or an internal network whose only way out is the egress proxy.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::core::traits::tools::Tool;
use crate::core::traits::tools::sandbox::Sandbox;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::sandbox::{
    SandboxError, SandboxOutput, SandboxRequest, session_name, workspace_path,
};
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::tools::structured;

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
    /// Size of the writable workspace, in mebibytes.
    pub workspace_mb: u32,
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
            workspace_mb: cfg.workspace_mb,
            egress: cfg.egress.then(|| Egress {
                network: cfg.egress_network.clone(),
                proxy_image: cfg.egress_proxy_image.clone(),
                proxy_name: cfg.egress_proxy_name.clone(),
            }),
        }
    }
}

/// Runs commands in a container. Clones share the session registry.
#[derive(Debug, Clone, Default)]
pub struct ContainerSandbox {
    limits: Limits,
    /// Container name to when it was last used, for the idle sweep and the per-caller cap.
    sessions: Arc<Mutex<HashMap<String, Instant>>>,
}

impl ContainerSandbox {
    /// Builds the sandbox over its limits.
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            sessions: Arc::default(),
        }
    }

    /// The prefix every session container of one caller shares.
    fn owner_prefix(ctx: &RequestContext) -> String {
        let mut hasher = DefaultHasher::new();
        (&ctx.tenant_id, &ctx.user_id).hash(&mut hasher);
        format!("sparky-sb-{:016x}-", hasher.finish())
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
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.remove(name);
        }
    }

    /// Records a session as used now.
    pub(crate) fn note_used(&self, name: &str) {
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.insert(name.to_owned(), Instant::now());
        }
    }

    /// The caller's least recently used session, when they already hold the most they may.
    pub(crate) fn least_recently_used(&self, ctx: &RequestContext) -> Option<String> {
        let prefix = Self::owner_prefix(ctx);
        let sessions = self.sessions.lock().ok()?;
        let mut held: Vec<(&String, &Instant)> = sessions
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .collect();
        if held.len() < self.limits.max_sessions.max(1) {
            return None;
        }
        held.sort_by_key(|(_, used)| **used);
        held.first().map(|(name, _)| (*name).clone())
    }

    /// The sessions idle past their budget.
    pub(crate) fn idle_sessions(&self) -> Vec<String> {
        let budget = Duration::from_secs(self.limits.session_idle_secs);
        let Ok(sessions) = self.sessions.lock() else {
            return Vec::new();
        };
        sessions
            .iter()
            .filter(|(_, used)| used.elapsed() >= budget)
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
    async fn start_session(&self, ctx: &RequestContext, name: &str) -> Result<(), SandboxError> {
        if self.running(name).await {
            self.note_used(name);
            return Ok(());
        }
        // A container under this name exists but has stopped; its workspace is already gone.
        self.remove(name).await;
        if let Some(lru) = self.least_recently_used(ctx) {
            self.remove(&lru).await;
        }
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
            self.note_used(name);
            return Ok(());
        }
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
        let session = request
            .session
            .as_deref()
            .map(session_name)
            .transpose()?
            .map(|name| Self::container_name(ctx, &name));
        let mut command = Command::new(&self.limits.runtime);
        match &session {
            Some(name) => {
                self.start_session(ctx, name).await?;
                command
                    .arg("exec")
                    .arg("--workdir")
                    .arg(WORKSPACE)
                    .arg(name);
            }
            None => {
                command.args(self.args());
            }
        }
        command
            .arg("sh")
            .arg("-c")
            .arg(&request.command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let budget = self.limits.timeout.min(ctx.remaining());
        let output = tokio::time::timeout(budget, command.output())
            .await
            .map_err(|_| SandboxError::Timeout)?
            .map_err(|e| {
                SandboxError::Runtime(format!("{} did not start: {e}", self.limits.runtime))
            })?;

        let max = self.limits.max_output_chars;
        Ok(SandboxOutput {
            // A process ended by a signal has no exit code and reports -1.
            exit_code: output.status.code().unwrap_or(-1),
            stdout: clip(&String::from_utf8_lossy(&output.stdout), max),
            stderr: clip(&String::from_utf8_lossy(&output.stderr), max),
            session: request.session.clone(),
        })
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
        self.start_session(ctx, &container).await?;
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
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "run_sandbox".into(),
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
