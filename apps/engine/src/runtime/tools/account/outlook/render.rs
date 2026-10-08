//! Rendering of the calendar and mail the Outlook tools hand back.

use std::fmt::Write as _;

use crate::core::types::tools::ToolOutput;
use crate::core::types::tools::outlook::{OutlookEvent, OutlookMessage};
use crate::runtime::tools::structured;

/// The calendar reply.
pub(super) fn calendar_output(events: Vec<OutlookEvent>, max: usize) -> ToolOutput {
    let shown: Vec<OutlookEvent> = events.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No upcoming Outlook events.".to_owned()
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
            let _ = write!(lines, "\n- {} {}{}{}", e.subject, when, place, link);
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}

/// The mail reply.
pub(super) fn mail_output(mail: Vec<OutlookMessage>, max: usize) -> ToolOutput {
    let shown: Vec<OutlookMessage> = mail.into_iter().take(max).collect();
    let text = if shown.is_empty() {
        "No recent Outlook mail.".to_owned()
    } else {
        let mut lines = format!("{} recent messages:", shown.len());
        for m in &shown {
            let from = m
                .from
                .as_deref()
                .map(|f| format!(" from {f}"))
                .unwrap_or_default();
            let when = m
                .received
                .as_deref()
                .map(|r| format!(" ({r})"))
                .unwrap_or_default();
            let preview = m
                .preview
                .as_deref()
                .map(|p| format!(" - {p}"))
                .unwrap_or_default();
            let _ = write!(lines, "\n- {}{}{}{}", m.subject, from, when, preview);
        }
        lines
    };
    ToolOutput {
        content: text,
        data: structured(&shown),
        sources: Vec::new(),
    }
}
