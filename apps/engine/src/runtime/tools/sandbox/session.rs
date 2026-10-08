//! Session bookkeeping: live containers, running commands, least recently used eviction, the reaper.

use std::process::Stdio;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use tokio::process::Command;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::sandbox::{SandboxCommand, SandboxError};
use crate::runtime::tools::sandbox::container::{
    ContainerSandbox, INSTANCE_LABEL, SESSION_PREFIX, held,
};

/// What the engine knows about one live session container.
#[derive(Debug, Clone)]
pub(super) struct Live {
    /// Name the caller gave it.
    pub(super) session: String,
    /// When it was started here.
    pub(super) started: Instant,
    /// When a command last ran in it.
    pub(super) used: Instant,
    /// Commands run in it.
    pub(super) runs: u64,
}

impl Live {
    pub(super) fn new(session: &str) -> Self {
        let now = Instant::now();
        Self {
            session: session.to_owned(),
            started: now,
            used: now,
            runs: 0,
        }
    }
}

/// Holds one command in the running list until the call that started it is dropped.
pub(super) struct Running {
    sandbox: ContainerSandbox,
    ticket: u64,
    started: Instant,
}

impl Running {
    /// Moves the finished command into the log and ends the guard.
    pub(super) fn ended(self, exit_code: Option<i32>, duration_ms: u64) {
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

impl ContainerSandbox {
    /// Records a command as started. The guard takes it out again however the call ends.
    pub(super) fn command_started(&self, mut command: SandboxCommand) -> Running {
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
    pub(super) fn take_running(&self, ticket: u64) -> Option<SandboxCommand> {
        let mut running = held(&self.running);
        let at = running.iter().position(|(t, _)| *t == ticket)?;
        Some(running.remove(at).1)
    }

    /// Puts an ended command in the log, dropping the oldest once the log is full.
    pub(super) fn keep(&self, command: SandboxCommand) {
        let mut recent = held(&self.recent);
        while recent.len() >= self.limits.recent_commands {
            recent.pop_front();
        }
        recent.push_back(command);
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

    /// Removes every session container of this instance the engine is not tracking.
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
    pub(super) async fn start_session(
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
