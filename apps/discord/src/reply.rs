//! Chunked replies, citation footers, confirmation prompts.

use std::fmt::Write;

use serenity::all::{ButtonStyle, CreateActionRow, CreateButton};

use crate::core::types::{ChatResponse, Confirmation};

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

/// The answer plus a citation footer, split into sendable messages.
pub fn render(resp: &ChatResponse) -> Vec<String> {
    let mut body = resp.text.trim().to_owned();
    if body.is_empty() {
        body = match resp.status.as_str() {
            "step_limit" => "I could not finish within the allowed number of steps.".into(),
            "deadline" => "That took too long; please try again.".into(),
            "cancelled" => "Cancelled.".into(),
            _ => "I have no answer for that.".into(),
        };
    }
    if let Some(c) = &resp.confirmation {
        let _ = write!(body, "\n\n**Needs your confirmation:** {}", c.summary);
    }
    if !resp.citations.is_empty() {
        body.push_str("\n\n**Sources**\n");
        for (i, c) in resp.citations.iter().enumerate() {
            let _ = writeln!(body, "{}. {c}", i + 1);
        }
    }
    chunk(&body, MAX_MESSAGE)
}

/// Confirm / cancel buttons bound to one confirmation token.
pub fn confirmation_row(c: &Confirmation) -> Vec<CreateActionRow> {
    vec![CreateActionRow::Buttons(vec![
        CreateButton::new(format!("confirm:{}", c.token))
            .style(ButtonStyle::Danger)
            .label(format!("Confirm {}", c.tool)),
        CreateButton::new(format!("cancel:{}", c.token))
            .style(ButtonStyle::Secondary)
            .label("Cancel"),
    ])]
}
