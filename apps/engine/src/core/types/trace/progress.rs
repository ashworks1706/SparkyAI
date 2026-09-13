//! Progress: the live progress event on the wire, and how much detail it carries.

use serde::Serialize;

use crate::core::types::trace::TraceEvent;

/// How much of a tool call, a result, or a thought one progress line shows.
#[derive(Debug, Clone, Copy)]
pub struct ProgressStyle {
    /// Characters kept from each piece of detail on a line.
    pub detail_chars: usize,
    /// Characters of reasoning a thinking line keeps.
    pub thought_chars: usize,
}

/// Characters of detail a progress line carries when nothing is configured.
pub const DETAIL_CHARS: usize = 160;

/// Characters of reasoning a thinking line carries when nothing is configured.
pub const THOUGHT_CHARS: usize = 600;

impl Default for ProgressStyle {
    fn default() -> Self {
        Self {
            detail_chars: DETAIL_CHARS,
            thought_chars: THOUGHT_CHARS,
        }
    }
}

/// One line of progress for a run's watcher. The engine renders text; a client just displays it.
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
    /// The text is the answer so far, not a step.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub draft: bool,
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
            draft: event.is_draft(),
            detail: event.clone(),
        })
    }
}
