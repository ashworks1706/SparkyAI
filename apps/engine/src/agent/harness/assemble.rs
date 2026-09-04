//! Context assembly. Fixed section order and per-section token budgets.
//!
//! Order: system instructions → role line → memory → evidence → history → current turn.
//! When over budget, evidence and history are trimmed before anything else; the system
//! prompt and the current turn are never dropped.

use crate::agent::harness::memory::Memory;
use crate::core::types::context::RequestContext;
use crate::core::types::evidence::Evidence;
use crate::core::types::message::{Message, Role};

/// Budgets for one assembled prompt, in estimated tokens.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Whole prompt, everything included.
    pub total: usize,
    /// Cap on the evidence section.
    pub evidence: usize,
    /// Cap on prior turns.
    pub history: usize,
    /// Cap on the memory section.
    pub memory: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            total: 3_000,
            evidence: 1_200,
            history: 1_000,
            memory: 300,
        }
    }
}

/// Everything that can go into a prompt.
#[derive(Debug, Default)]
pub struct Sections<'a> {
    /// Versioned system instructions.
    pub system: &'a str,
    /// Recalled memories, best first.
    pub memory: &'a [Memory],
    /// Retrieved evidence, best first.
    pub evidence: &'a [Evidence],
    /// Prior turns and this request's own tool exchanges, oldest first.
    pub history: &'a [Message],
    /// The user's current message.
    pub input: &'a str,
}

/// The assembled prompt plus what was left out.
#[derive(Debug)]
pub struct Assembled {
    /// Messages to send, in order.
    pub messages: Vec<Message>,
    /// Rough token estimate of the whole thing.
    pub estimated_tokens: usize,
    /// Evidence chunks that made it in.
    pub evidence_used: usize,
    /// History turns that made it in.
    pub history_used: usize,
}

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
    if !s.evidence.is_empty() {
        let mut block = String::from(
            "Evidence from ASU sources. Answer only from this; cite sources by number. \
             If it does not answer the question, say so.\n",
        );
        let mut spent = estimate(&block);
        for (i, e) in s.evidence.iter().enumerate() {
            let entry = format!(
                "\n[{}] {} (fetched {})\n{}\n",
                i + 1,
                e.title,
                e.fetched_at.format("%Y-%m-%d"),
                e.content.trim()
            );
            let cost = estimate(&entry);
            if spent + cost > budget.evidence {
                break;
            }
            block.push_str(&entry);
            spent += cost;
            evidence_used += 1;
        }
        used += spent;
        messages.push(Message::system(block));
    }

    let input_cost = estimate(s.input);
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::Utc;
    use uuid::Uuid;

    use super::*;

    fn ctx() -> RequestContext {
        RequestContext::new("g", "u", Duration::from_secs(5))
    }

    fn evidence(n: usize) -> Vec<Evidence> {
        (0..n)
            .map(|i| Evidence {
                source_id: Uuid::new_v4(),
                chunk_id: Uuid::new_v4(),
                title: format!("Doc {i}"),
                content: "x".repeat(400),
                url: None,
                fetched_at: Utc::now(),
                score: 1.0,
            })
            .collect()
    }

    #[test]
    fn system_comes_first_and_input_last() {
        let out = assemble(
            &ctx(),
            &Sections {
                system: "You are Sparky.",
                input: "hi",
                ..Sections::default()
            },
            Budget::default(),
        );
        assert_eq!(out.messages.first().map(|m| m.role), Some(Role::System));
        assert_eq!(out.messages.last().map(|m| m.content.as_str()), Some("hi"));
    }

    #[test]
    fn evidence_is_trimmed_to_its_budget() {
        let ev = evidence(20);
        let out = assemble(
            &ctx(),
            &Sections {
                system: "s",
                evidence: &ev,
                input: "q",
                ..Sections::default()
            },
            Budget {
                evidence: 500,
                ..Budget::default()
            },
        );
        assert!(out.evidence_used < 20);
        assert!(out.evidence_used >= 1);
    }

    #[test]
    fn history_keeps_the_newest_turns() {
        let history: Vec<Message> = (0..50)
            .map(|i| Message::user(format!("turn {i} {}", "y".repeat(100))))
            .collect();
        let out = assemble(
            &ctx(),
            &Sections {
                system: "s",
                history: &history,
                input: "q",
                ..Sections::default()
            },
            Budget {
                history: 300,
                ..Budget::default()
            },
        );
        assert!(out.history_used < 50);
        let last_history = &out.messages[out.messages.len() - 2];
        assert!(last_history.content.starts_with("turn 49"));
    }

    #[test]
    fn history_never_starts_with_a_tool_result() {
        let history = vec![
            Message::assistant_tool_calls("", vec![]),
            Message::tool_result("c1", "big result ".repeat(50)),
            Message::user("follow-up"),
        ];
        let out = assemble(
            &ctx(),
            &Sections {
                system: "s",
                history: &history,
                input: "q",
                ..Sections::default()
            },
            Budget {
                history: 160,
                ..Budget::default()
            },
        );
        let first_history = out.messages.iter().skip(1).find(|m| m.role != Role::System);
        assert!(first_history.is_none_or(|m| m.role != Role::Tool));
    }
}
