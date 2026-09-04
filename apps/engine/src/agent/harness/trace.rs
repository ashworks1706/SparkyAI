//! JSONL trace sink: one file per request, one event per line.

use std::path::PathBuf;

use chrono::Utc;
use uuid::Uuid;

use crate::core::traits::trace::TraceSink;
use crate::core::types::context::RequestContext;
use crate::core::types::trace::{TraceEvent, TraceRecord};

fn record(ctx: &RequestContext, event: TraceEvent) -> TraceRecord {
    TraceRecord {
        request_id: ctx.request_id,
        conversation_id: ctx.conversation_id,
        at: Utc::now(),
        event,
    }
}

/// Appends one JSON line per event to `<dir>/<request_id>.jsonl`.
#[derive(Debug)]
pub struct JsonlSink {
    dir: PathBuf,
}

impl JsonlSink {
    /// Writes under `dir`, creating it if missing.
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Path of the trace file for a request.
    pub fn path_for(&self, request_id: Uuid) -> PathBuf {
        self.dir.join(format!("{request_id}.jsonl"))
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
            .and_then(|mut f| writeln!(f, "{line}"));
        if let Err(e) = result {
            tracing::warn!(error = %e, path = %path.display(), "trace write failed");
        }
    }
}
