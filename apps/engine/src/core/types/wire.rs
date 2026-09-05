//! Wire shapes: tool argument schemas and the live progress event.

use serde::{Deserialize, Serialize};

use crate::core::types::trace::TraceEvent;

/// Arguments the model passes to `search_asu`.
#[derive(Deserialize)]
pub struct SearchArgs {
    /// What to look for.
    pub query: String,
    /// Optional source categories to restrict to.
    #[serde(default)]
    pub categories: Vec<String>,
}

/// One line of progress for whoever is watching a run.
///
/// `text` is rendered by the engine so a client can display any event, including kinds added
/// after the client was written. `event` names the kind for clients that special-case one, and
/// `detail` carries the event itself for anything richer.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    /// Snake-case name of the trace event this came from.
    pub event: &'static str,
    /// Ready-to-display sentence.
    pub text: String,
    /// The event itself, for clients that want more than `text`.
    pub detail: TraceEvent,
}

impl Progress {
    /// The progress line for an event, or `None` when the event is bookkeeping.
    pub fn of(event: &TraceEvent) -> Option<Self> {
        Some(Self {
            event: event.kind(),
            text: event.progress()?,
            detail: event.clone(),
        })
    }
}
