//! Trace sink behaviour: JSONL on disk, in-memory, fan-out.

use std::path::PathBuf;

use chrono::Utc;
use uuid::Uuid;

use crate::core::traits::trace::TraceSink;
use crate::core::types::context::RequestContext;
use crate::core::types::harness::{JsonlSink, MemorySink, MultiSink};
use crate::core::types::trace::{TraceEvent, TraceRecord};

impl TraceSink for () {
    fn emit(&self, _ctx: &RequestContext, _event: TraceEvent) {}
}

fn record(ctx: &RequestContext, event: TraceEvent) -> TraceRecord {
    TraceRecord {
        request_id: ctx.request_id,
        conversation_id: ctx.conversation_id,
        at: Utc::now(),
        event,
    }
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

    /// Reads every record of one request, in order.
    pub fn read(&self, request_id: Uuid) -> std::io::Result<Vec<TraceRecord>> {
        let text = std::fs::read_to_string(self.path_for(request_id))?;
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(std::io::Error::other))
            .collect()
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

impl MemorySink {
    /// Everything emitted so far.
    pub fn records(&self) -> Vec<TraceRecord> {
        self.records.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

impl TraceSink for MemorySink {
    fn emit(&self, ctx: &RequestContext, event: TraceEvent) {
        if let Ok(mut records) = self.records.lock() {
            records.push(record(ctx, event));
        }
    }
}

impl TraceSink for MultiSink {
    fn emit(&self, ctx: &RequestContext, event: TraceEvent) {
        for sink in &self.0 {
            sink.emit(ctx, event.clone());
        }
    }
}
