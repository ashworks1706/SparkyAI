//! Progress: the live progress event on the wire.

use serde::Serialize;

use crate::core::types::trace::TraceEvent;

/// One line of progress for whoever is watching a run.
///
/// text is rendered by the engine; a client can display any event, including kinds added after
/// the client was written. event names the kind, and detail carries the event itself.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    /// Snake-case name of the trace event this came from.
    pub event: &'static str,
    /// Ready-to-display sentence.
    pub text: String,
    /// The event itself, for clients that want more than text.
    pub detail: TraceEvent,
}

impl Progress {
    /// The progress line for an event, or None when the event is bookkeeping.
    pub fn of(event: &TraceEvent) -> Option<Self> {
        Some(Self {
            event: event.kind(),
            text: event.progress()?,
            detail: event.clone(),
        })
    }
}
