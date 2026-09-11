//! A command in an isolated environment, and the tool that offers it.
//!
//! The container runs with no network, a read-only root, a memory and process ceiling, and a
//! non-root user. Nothing it does reaches the host, the database, or the model endpoint.
//! Sessions are not kept: every call starts from the image.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::core::traits::sandbox::Sandbox;
use crate::core::traits::tool::Tool;
use crate::core::types::context::RequestContext;
use crate::core::types::sandbox::{SandboxError, SandboxOutput, SandboxRequest};
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};

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
        }
    }
}

/// Runs commands in a container.
#[derive(Debug, Clone, Default)]
pub struct ContainerSandbox {
    limits: Limits,
}

impl ContainerSandbox {
    /// Builds the sandbox over its limits.
    pub fn new(limits: Limits) -> Self {
        Self { limits }
    }

    /// The arguments that seal the container. Every one of them is load bearing.
    pub fn args(&self) -> Vec<String> {
        let l = &self.limits;
        [
            "run",
            "--rm",
            "--network",
            "none",
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
            "/tmp:rw,noexec,nosuid,size=64m",
            "--workdir",
            "/tmp",
            &l.image,
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
}

/// Keeps the first max chars, marking what was dropped.
fn clip(text: &str, max: usize) -> String {
    if max == 0 || text.chars().count() <= max {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max).collect();
    format!("{kept}\n[truncated]")
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
        let mut command = Command::new(&self.limits.runtime);
        command
            .args(self.args())
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
            exit_code: output.status.code().unwrap_or(-1),
            stdout: clip(&String::from_utf8_lossy(&output.stdout), max),
            stderr: clip(&String::from_utf8_lossy(&output.stderr), max),
        })
    }
}

/// Offers the sandbox to the model.
pub struct SandboxTool {
    sandbox: Arc<dyn Sandbox>,
    risk: RiskClass,
}

impl SandboxTool {
    /// Builds the tool. The risk class it declares is what Policy gates it by.
    pub fn new(sandbox: Arc<dyn Sandbox>, risk: RiskClass) -> Self {
        Self { sandbox, risk }
    }
}

#[async_trait]
impl Tool for SandboxTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "run_sandbox".into(),
            description: "Run a shell command in an isolated environment to compute or reshape \
                          data you already have. There is no network and no filesystem beyond a \
                          temporary directory, so it cannot fetch anything or reach ASU."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The shell command to run." }
                },
                "required": ["command"]
            }),
            risk: self.risk,
            // Runs are independent, but a container is expensive enough that two at once on one
            // step is worth avoiding.
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
            data: serde_json::to_value(&out).ok(),
        })
    }
}
