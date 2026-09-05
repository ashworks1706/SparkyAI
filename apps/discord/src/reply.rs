//! Chunked replies and citation footers.

use std::fmt::Write;

use crate::core::types::{ChatResponse, EngineError};

/// Discord's hard limit on message content.
pub const MAX_MESSAGE: usize = 2_000;

/// Splits `text` into messages of at most `limit` bytes on line, then space, boundaries.
pub fn chunk(text: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text.trim();
    while !rest.is_empty() {
        if rest.len() <= limit {
            out.push(rest.to_owned());
            break;
        }
        let mut cut = limit;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        let window = &rest[..cut];
        let split = window
            .rfind('\n')
            .or_else(|| window.rfind(' '))
            .filter(|&i| i > limit / 4)
            .unwrap_or(cut);
        out.push(rest[..split].trim_end().to_owned());
        rest = rest[split..].trim_start();
    }
    out
}

/// What to say when the engine call failed. Capacity is temporary and self-clearing; anything
/// else is an outage.
pub fn failure(e: &EngineError) -> String {
    match e {
        EngineError::Status { status: 503, .. } => {
            "Sparky is busy with other questions right now. Try again in a moment.".into()
        }
        EngineError::Status { .. } | EngineError::Transport(_) => {
            "Sparky is unavailable right now. Please try again shortly.".into()
        }
    }
}

/// The answer plus a citation footer, split into sendable messages.
pub fn render(resp: &ChatResponse) -> Vec<String> {
    let mut body = resp.text.trim().to_owned();
    if body.is_empty() {
        body = match resp.status.as_str() {
            "step_limit" => "I could not finish within the allowed number of steps.".into(),
            "stalled" => "I kept repeating myself without getting further; try rephrasing.".into(),
            "deadline" => "That took too long; please try again.".into(),
            "cancelled" => "Cancelled.".into(),
            _ => "I have no answer for that.".into(),
        };
    }
    if let Some(c) = &resp.confirmation {
        // The engine stopped before acting. Relaying an approval needs the Phase 3 confirm
        // endpoint; until then the bot says so rather than showing buttons that do nothing.
        let _ = write!(
            body,
            "\n\n**Not done — `{}` needs your approval:** {} Approving actions from Discord \
             is not available yet.",
            c.tool, c.summary
        );
    }
    if !resp.citations.is_empty() {
        body.push_str("\n\n**Sources**\n");
        for (i, c) in resp.citations.iter().enumerate() {
            let _ = writeln!(body, "{}. {c}", i + 1);
        }
    }
    chunk(&body, MAX_MESSAGE)
}
