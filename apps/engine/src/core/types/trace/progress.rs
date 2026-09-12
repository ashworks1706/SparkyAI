//! Progress: the live progress event on the wire, and how much detail it carries.

use serde::Serialize;

use crate::core::types::trace::TraceEvent;

/// How much of a tool call, a result, or a thought one progress line shows.
#[derive(Debug, Clone, Copy)]
pub struct ProgressStyle {
    /// Characters kept from each piece of detail on a line.
    pub detail_chars: usize,
}

/// Characters of detail a progress line carries when nothing is configured.
pub const DETAIL_CHARS: usize = 160;

impl Default for ProgressStyle {
    fn default() -> Self {
        Self {
            detail_chars: DETAIL_CHARS,
        }
    }
}

/// One line of progress for whoever is watching a run.
///
/// text is rendered by the engine; a client can display any event, including kinds added after
/// the client was written. event names the kind, detail carries the event itself, slot names
/// the line this one writes over, and clear takes that line away.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    /// Snake-case name of the trace event this came from.
    pub event: &'static str,
    /// Ready-to-display sentence.
    pub text: String,
    /// The line this replaces, when it replaces one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slot: Option<String>,
    /// Removes the line of slot instead of writing text there.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub clear: bool,
    /// The event itself, for clients that want more than text.
    pub detail: TraceEvent,
}

impl Progress {
    /// The progress line for an event, or None when the event is bookkeeping.
    pub fn of(event: &TraceEvent, style: ProgressStyle) -> Option<Self> {
        let slot = event.slot();
        let clear = event.clears_slot() && slot.is_some();
        let text = event.progress(style);
        if text.is_none() && !clear {
            return None;
        }
        Some(Self {
            event: event.kind(),
            text: text.unwrap_or_default(),
            slot,
            clear,
            detail: event.clone(),
        })
    }
}
