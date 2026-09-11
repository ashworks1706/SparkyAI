//! Context assembly. Fixed section order and per-section token budgets.
//!
//! Order: system instructions, role line, memory, evidence, history, current turn.
//! When over budget, evidence and history are trimmed first. The system prompt and the
//! current turn are never dropped.

use crate::core::types::agent::assemble::{Assembled, Budget, Sections};
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::message::{Message, Role};
use crate::core::types::model::tokens::estimate;

/// Builds the message list within budget.
pub fn assemble(ctx: &RequestContext, s: &Sections<'_>, budget: Budget) -> Assembled {
    let mut messages = Vec::new();
    let mut used = 0usize;

    let cpt = budget.chars_per_token;
    let role_line = if ctx.roles.is_empty() {
        s.templates
            .role_line_no_roles
            .replace("{user}", &ctx.user_id)
    } else {
        s.templates
            .role_line
            .replace("{user}", &ctx.user_id)
            .replace("{roles}", &ctx.roles.join(", "))
    };
    let mut system = format!("{}\n\n{role_line}", s.system.trim());
    if !s.date.is_empty() {
        let date_line = s.templates.date_line.replace("{date}", s.date);
        system.push('\n');
        system.push_str(&date_line);
    }
    used += estimate(&system, cpt);
    messages.push(Message::system(system));

    if !s.capabilities.trim().is_empty() {
        let block = s.capabilities.trim().to_owned();
        let cost = estimate(&block, cpt);
        if cost <= budget.capabilities {
            used += cost;
            messages.push(Message::system(block));
        }
    }

    let mut memory_used = 0;
    if let Some((block, spent, count)) = memory_block(s, budget.memory, cpt) {
        used += spent;
        memory_used = count;
        messages.push(Message::system(block));
    }

    let mut evidence_used = 0;
    let input_cost = estimate(s.input, cpt);
    if !s.evidence.is_empty() {
        let mut block = format!("{}\n", s.templates.evidence_header.trim());
        let mut spent = estimate(&block, cpt);
        // Evidence is capped by its own budget and by what remains of the total after the
        // sections above and the current input.
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
            let cost = estimate(&entry, cpt);
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
        let cost = m.estimated_tokens(cpt);
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
        memory_used,
    }
}

/// The memory section: the header and every memory that fits budget, with the tokens it
/// spends and how many memories it holds. None when there is no memory.
fn memory_block(s: &Sections<'_>, budget: usize, cpt: usize) -> Option<(String, usize, usize)> {
    if s.memory.is_empty() {
        return None;
    }
    let mut block = format!("{}\n", s.templates.memory_header.trim());
    let mut spent = estimate(&block, cpt);
    let mut count = 0;
    for m in s.memory {
        let line = format!("- ({}) {}\n", m.kind.as_str(), m.content.trim());
        let cost = estimate(&line, cpt);
        if spent + cost > budget {
            break;
        }
        block.push_str(&line);
        spent += cost;
        count += 1;
    }
    Some((block, spent, count))
}
