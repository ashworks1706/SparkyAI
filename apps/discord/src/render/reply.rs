//! Chunked replies, failure lines, and the text of the memory commands.

use std::fmt::Write;

use crate::core::types::{EngineError, ProfileList};

/// The Discord hard limit on message content. bot.max_message_chars may go below it, never
/// above.
pub const MAX_MESSAGE: usize = 2_000;

/// Splits text into messages of at most limit bytes on line, then space, boundaries.
pub fn chunk(text: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text.trim();
    while !rest.is_empty() {
        if rest.len() <= limit {
            out.push(rest.to_owned());
            break;
        }
        let mut cut = limit;
        while cut > 0 && !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        if cut == 0 {
            cut = rest.chars().next().map_or(rest.len(), char::len_utf8);
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

/// What to say when Sparky cannot be reached or the request could not be served.
pub const UNAVAILABLE: &str = "Sparky is unavailable right now. Please try again shortly.";

/// What to say when the engine call failed. Capacity is temporary and self-clearing. Anything
/// else is an outage.
pub fn failure(e: &EngineError) -> String {
    match e {
        EngineError::Status { status: 503, .. } => {
            "Sparky is busy with other questions right now. Try again in a moment.".into()
        }
        EngineError::Status { status: 409, .. } => "That approval is no longer open.".into(),
        EngineError::Status { status: 404, .. } => {
            "I lost track of that conversation. Ask again to start a new one.".into()
        }
        EngineError::Status { .. } | EngineError::Transport(_) => UNAVAILABLE.into(),
    }
}

/// What /memory says when the engine remembers nothing.
pub const NOTHING_REMEMBERED: &str = "I do not remember anything about you yet.";

/// What a presser sees when the engine refused or failed their approval press.
pub fn confirm_failure(e: &EngineError) -> String {
    if confirm_refused(e) {
        NOT_YOURS.into()
    } else {
        failure(e)
    }
}

/// What a presser sees when the approval is not theirs or no longer open.
pub const NOT_YOURS: &str = "This approval is not yours, or it is no longer open.";

/// Whether the engine refused an approval press: not the asker, or no longer open.
pub fn confirm_refused(e: &EngineError) -> bool {
    matches!(
        e,
        EngineError::Status {
            status: 404 | 409,
            ..
        }
    )
}

/// What remembered things and relations look like, split into sendable messages.
pub fn render_profile(profile: &ProfileList, limit: usize) -> Vec<String> {
    if profile.nodes.is_empty() && profile.relations.is_empty() {
        return vec![NOTHING_REMEMBERED.to_owned()];
    }
    let mut body = String::new();
    if !profile.nodes.is_empty() {
        body.push_str("**What I remember**\n");
        for n in &profile.nodes {
            let _ = writeln!(
                body,
                "- {} ({}, {})",
                n.label,
                n.kind,
                percent(n.confidence)
            );
        }
    }
    if !profile.relations.is_empty() {
        if !body.is_empty() {
            body.push('\n');
        }
        body.push_str("**How it connects**\n");
        for r in &profile.relations {
            let _ = writeln!(
                body,
                "- {} {} {} ({})",
                r.subject,
                r.relation.replace('_', " "),
                r.object,
                percent(r.confidence)
            );
        }
    }
    body.push_str("\n/forget with a label drops one thing. /forget alone clears everything.");
    chunk(&body, limit.clamp(1, MAX_MESSAGE))
}

/// A confidence from 0 to 1 as a whole percentage.
fn percent(confidence: f64) -> String {
    format!("{:.0}%", confidence.clamp(0.0, 1.0) * 100.0)
}

/// What /forget says after the engine removed some items. named is true for a single label.
pub fn forgot(removed: u64, named: bool) -> String {
    match (removed, named) {
        (0, true) => "I had nothing under that name.".into(),
        (0, false) => "I did not remember anything about you.".into(),
        (1, _) => "Forgot 1 thing.".into(),
        (n, _) => format!("Forgot {n} things."),
    }
}

/// What to say when a profile call failed. 503 means the profile graph is off.
pub fn memory_failure(e: &EngineError) -> String {
    match e {
        EngineError::Status { status: 503, .. } => "Memory is turned off here.".into(),
        EngineError::Status { status: 404, .. } => UNAVAILABLE.into(),
        other => failure(other),
    }
}
