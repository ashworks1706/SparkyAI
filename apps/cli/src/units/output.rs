//! Reading what child processes print: plain text lines and docker compose ps rows.

use std::collections::HashMap;

use crate::core::types::{ComposePsRow, ServiceState};

/// Removes ANSI escape sequences and every control character except tab.
pub fn sanitize_line(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for d in chars.by_ref() {
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        // Tabs are the only control character the pane keeps.
        if c.is_control() && c != '\t' {
            continue;
        }
        out.push(c);
    }
    out
}

/// Parses docker compose ps json output, as an array or one object per line.
pub fn parse_ps(raw: &str) -> Result<HashMap<String, ServiceState>, String> {
    let rows: Vec<ComposePsRow> = if raw.trim().is_empty() {
        Vec::new()
    } else if let Ok(rows) = serde_json::from_str::<Vec<ComposePsRow>>(raw) {
        rows
    } else {
        raw.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                serde_json::from_str::<ComposePsRow>(l).map_err(|e| format!("compose ps row: {e}"))
            })
            .collect::<Result<_, _>>()?
    };
    Ok(rows
        .into_iter()
        .map(|r| {
            (
                r.service,
                ServiceState {
                    state: r.state,
                    health: r.health,
                    exit_code: r.exit_code,
                },
            )
        })
        .collect())
}
