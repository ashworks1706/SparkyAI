//! Context assembly. Fixed section order and per-section token budgets.
//!
//! Order: system instructions → role line → memory → evidence → history → current turn.
//! When over budget, evidence and history are trimmed before anything else; the system
//! prompt and the current turn are never dropped.

use crate::core::types::assemble::{Assembled, Budget, Sections};
use crate::core::types::context::RequestContext;
use crate::core::types::message::{Message, Role};

fn estimate(text: &str) -> usize {
    text.len() / 4 + 4
}

/// Builds the message list within `budget`.
pub fn assemble(ctx: &RequestContext, s: &Sections<'_>, budget: Budget) -> Assembled {
    let mut messages = Vec::new();
    let mut used = 0usize;

    let role_line = if ctx.roles.is_empty() {
        format!("The user is `{}`. They hold no special roles.", ctx.user_id)
    } else {
        format!(
            "The user is `{}`. Roles: {}.",
            ctx.user_id,
            ctx.roles.join(", ")
        )
    };
    let system = format!("{}\n\n{role_line}", s.system.trim());
    used += estimate(&system);
    messages.push(Message::system(system));

    if !s.memory.is_empty() {
        let mut block = String::from("What you remember about this user:\n");
        let mut spent = estimate(&block);
        for m in s.memory {
            let line = format!("- ({}) {}\n", m.kind.as_str(), m.content.trim());
            let cost = estimate(&line);
            if spent + cost > budget.memory {
                break;
            }
            block.push_str(&line);
            spent += cost;
        }
        used += spent;
        messages.push(Message::system(block));
    }

    let mut evidence_used = 0;
    let input_cost = estimate(s.input);
    if !s.evidence.is_empty() {
        let mut block = String::from(
            "Evidence from ASU sources. Answer only from this; cite sources by number. \
             If it does not answer the question, say so.\n",
        );
        let mut spent = estimate(&block);
        // Evidence never eats the whole prompt: it is capped by its own budget and by what
        // remains of the total after the sections above and the current input.
        let evidence_budget = budget
            .evidence
            .min(budget.total.saturating_sub(used + input_cost));
        for (i, e) in s.evidence.iter().enumerate() {
            let entry = format!(
                "\n[{}] {} (fetched {})\n{}\n",
                i + 1,
                e.title,
                e.fetched_at.format("%Y-%m-%d"),
                e.content.trim()
            );
            let cost = estimate(&entry);
            if spent + cost > evidence_budget {
                break;
            }
            block.push_str(&entry);
            spent += cost;
            evidence_used += 1;
        }
        used += spent;
        messages.push(Message::system(block));
    }

    let remaining_total = budget.total.saturating_sub(used + input_cost);
    let history_budget = budget.history.min(remaining_total);
    let mut kept: Vec<&Message> = Vec::new();
    let mut spent = 0usize;
    for m in s.history.iter().rev() {
        let cost = m.estimated_tokens();
        if spent + cost > history_budget {
            break;
        }
        kept.push(m);
        spent += cost;
    }
    // Never start history with an orphaned tool result.
    while kept.last().is_some_and(|m| m.role == Role::Tool) {
        kept.pop();
    }
    kept.reverse();
    let history_used = kept.len();
    used += spent;
    messages.extend(kept.into_iter().cloned());

    if !s.input.is_empty() {
        used += input_cost;
        messages.push(Message::user(s.input));
    }

    Assembled {
        messages,
        estimated_tokens: used,
        evidence_used,
        history_used,
    }
}
