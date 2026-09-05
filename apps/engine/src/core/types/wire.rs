//! Wire shapes: tool argument schemas and the live progress event.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
/// `detail` carries the event's own fields for anything richer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    /// Snake-case name of the trace event this came from.
    pub event: String,
    /// Ready-to-display sentence.
    pub text: String,
    /// The event's own fields, for clients that want more than `text`.
    pub detail: Value,
}

impl Progress {
    /// The progress line for an event, or `None` when the event is bookkeeping.
    pub fn of(event: &TraceEvent) -> Option<Self> {
        let text = event.progress()?;
        let detail = serde_json::to_value(event).unwrap_or(Value::Null);
        let name = detail
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        Some(Self {
            event: name,
            text,
            detail,
        })
    }
}
