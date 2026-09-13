//! JSONL trace sink: one file per request, one event per line.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use chrono::Utc;
use uuid::Uuid;

use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::trace::progress::{Progress, ProgressStyle};
use crate::core::types::trace::{TraceEvent, TraceRecord};

fn record(ctx: &RequestContext, event: TraceEvent) -> TraceRecord {
    TraceRecord {
        request_id: ctx.request_id,
        conversation_id: ctx.conversation_id,
        at: Utc::now(),
        event,
    }
}

/// Appends one JSON line per event to a file named for the request id, under the trace dir.
#[derive(Debug)]
pub struct JsonlSink {
    dir: PathBuf,
    /// Stop writing the trace of a request past this many bytes. Zero removes the limit.
    max_file_bytes: u64,
}

impl JsonlSink {
    /// Writes under dir, creating it if missing. max_file_bytes caps the trace of one request.
    /// Zero removes the limit.
    pub fn new(dir: impl Into<PathBuf>, max_file_bytes: u64) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            max_file_bytes,
        })
    }

    /// Path of the trace file for a request.
    pub fn path_for(&self, request_id: Uuid) -> PathBuf {
        self.dir.join(format!("{request_id}.jsonl"))
    }

    /// Deletes trace files last modified more than older_than ago. Returns how many were
    /// removed. Called once at boot.
    pub fn prune(&self, older_than: Duration) -> std::io::Result<usize> {
        let cutoff = SystemTime::now()
            .checked_sub(older_than)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let mut removed = 0;
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry.path().extension().is_none_or(|e| e != "jsonl") {
                continue;
            }
            let stale = entry
                .metadata()
                .and_then(|m| m.modified())
                .is_ok_and(|modified| modified < cutoff);
            if stale && std::fs::remove_file(entry.path()).is_ok() {
                removed += 1;
            }
        }
        Ok(removed)
    }
}

impl TraceSink for JsonlSink {
    fn emit(&self, ctx: &RequestContext, event: TraceEvent) {
        use std::io::Write;
        let Ok(line) = serde_json::to_string(&record(ctx, event)) else {
            return;
        };
        let path = self.path_for(ctx.request_id);
        let result = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| {
                // The cap drops the tail of the trace, keeping the start of the run.
                if self.max_file_bytes > 0 && f.metadata()?.len() >= self.max_file_bytes {
                    return Ok(());
                }
                writeln!(f, "{line}")
            });
        if let Err(e) = result {
            tracing::warn!(error = %e, path = %path.display(), "trace write failed");
        }
    }
}

/// Discards every event. Used when trace recording is switched off.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl TraceSink for NullSink {
    fn emit(&self, _ctx: &RequestContext, _event: TraceEvent) {}
}

/// Records every event through the sink beneath it and, when the caller is watching, forwards
/// the ones that map to progress.
pub struct Fanout {
    inner: Arc<dyn TraceSink>,
    style: ProgressStyle,
}

impl Fanout {
    /// Wraps the sink that records the full trace, with the default amount of detail.
    pub fn new(inner: Arc<dyn TraceSink>) -> Self {
        Self {
            inner,
            style: ProgressStyle::default(),
        }
    }

    /// Sets how much detail a forwarded progress line carries.
    #[must_use]
    pub fn with_style(mut self, style: ProgressStyle) -> Self {
        self.style = style;
        self
    }
}

impl TraceSink for Fanout {
    fn emit(&self, ctx: &RequestContext, event: TraceEvent) {
        if let Some(tx) = &ctx.progress
            && let Some(progress) = Progress::of(&event, self.style)
        {
            // A dropped receiver means the caller stopped watching. The trace still lands.
            let _ = tx.send(progress);
        }
        self.inner.emit(ctx, event);
    }
}
