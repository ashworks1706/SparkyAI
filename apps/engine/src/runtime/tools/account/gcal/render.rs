//! Rendering of the events the Google Calendar tool hands back.

use std::fmt::Write as _;

use crate::core::types::tools::ToolOutput;
use crate::core::types::tools::gcal::GCalEvent;
use crate::runtime::tools::structured;

/// The calendar reply.
pub(super) fn output(events: Vec<GCalEvent>, max: usize) -> ToolOutput {
    let shown: Vec<GCalEvent> = events.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No upcoming Google Calendar events.".to_owned()
    } else {
        let mut lines = format!("{} upcoming events:", shown.len());
        for e in &shown {
            let when = e.start.as_deref().unwrap_or("no start time");
            let place = e
                .location
                .as_deref()
                .map(|l| format!(" at {l}"))
                .unwrap_or_default();
            let link = e
                .url
                .as_deref()
                .map(|u| format!(" {u}"))
                .unwrap_or_default();
            let _ = write!(lines, "\n- {} {}{}{}", e.summary, when, place, link);
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}
