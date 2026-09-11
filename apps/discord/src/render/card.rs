//! The one message a turn lives in: steps while it runs, then the answer and its footers.

use std::fmt::Write;
use std::time::{Duration, Instant};

use crate::core::types::ChatResponse;
use crate::render::reply::{MAX_MESSAGE, chunk};

/// Header of the message while a turn runs.
pub const THINKING: &str = "**Sparky is thinking…**";

/// Prefix of one step line.
const BULLET: &str = "-# • ";

/// Items each footer keeps once footers are trimmed to fit.
const FOOTER_KEEP: usize = 3;

/// The steps the engine reported for one turn, in order.
#[derive(Debug, Default, Clone)]
pub struct Steps {
    lines: Vec<String>,
}

impl Steps {
    /// Appends a step and says whether the list changed. A blank line or an immediate repeat
    /// is dropped.
    pub fn push(&mut self, text: &str) -> bool {
        let text = text.trim();
        if text.is_empty() || self.lines.last().is_some_and(|last| last == text) {
            return false;
        }
        self.lines.push(text.to_owned());
        true
    }

    /// The steps so far.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }
}

/// Paces edits of the running card: at most one per interval, and a change inside the gap
/// waits for the next slot instead of being dropped.
#[derive(Debug, Clone)]
pub struct Pacer {
    every: Duration,
    last: Instant,
    dirty: bool,
}

impl Pacer {
    /// A pacer whose last edit happened at now.
    pub fn new(every: Duration, now: Instant) -> Self {
        Self {
            every,
            last: now,
            dirty: false,
        }
    }

    /// Records whether the card content changed.
    pub fn mark(&mut self, changed: bool) {
        self.dirty |= changed;
    }

    /// How long until the pending change is due, or None when nothing waits.
    pub fn wait(&self, now: Instant) -> Option<Duration> {
        self.dirty.then(|| {
            self.every
                .saturating_sub(now.saturating_duration_since(self.last))
        })
    }

    /// Records an edit made at now.
    pub fn flushed(&mut self, now: Instant) {
        self.dirty = false;
        self.last = now;
    }
}

/// The message while the turn runs. The oldest steps fold into a count when they do not fit.
pub fn thinking(steps: &[String], limit: usize) -> String {
    let limit = clamp(limit);
    for hidden in 0..=steps.len() {
        let mut out = THINKING.to_owned();
        if hidden > 0 {
            let _ = write!(out, "\n-# {hidden} earlier {}", plural(hidden));
        }
        for step in steps.iter().skip(hidden) {
            out.push('\n');
            out.push_str(&bullet(step));
        }
        if out.len() <= limit {
            return out;
        }
    }
    fit(THINKING.to_owned(), limit)
}

/// The finished turn: steps, answer, then the footers that have content. Steps fold into a
/// count first, footers trim next, and only an answer that still does not fit spills into
/// continuation messages.
pub fn answer(steps: &[String], resp: &ChatResponse, limit: usize) -> Vec<String> {
    layout(0, steps, resp, limit)
}

/// The card after an accepted approval: the steps of the old card, a step saying what was
/// decided, then the resumed answer and its footers. The old prompt and footers are dropped.
pub fn resumed(old: &str, approved: bool, resp: &ChatResponse, limit: usize) -> Vec<String> {
    let (folded, mut steps) = steps_of(old);
    steps.push(if approved { "approved" } else { "declined" }.to_owned());
    layout(folded, &steps, resp, limit)
}

/// The turn that failed: steps, then the failure line.
pub fn failed(steps: &[String], line: &str, limit: usize) -> String {
    let limit = clamp(limit);
    let full = compose(&[&bullets(steps), line]);
    if full.len() <= limit {
        return full;
    }
    fit(compose(&[&count_line(steps.len()), line]), limit)
}

/// The steps a rendered card shows: how many were folded into a count, and the bullets.
pub fn steps_of(card: &str) -> (usize, Vec<String>) {
    let mut folded = 0;
    let mut steps = Vec::new();
    for line in card.lines() {
        if let Some(step) = line.strip_prefix(BULLET) {
            steps.push(step.to_owned());
        } else if let Some(rest) = line.strip_prefix("-# ")
            && let Some(n) = rest
                .strip_suffix(" steps")
                .or_else(|| rest.strip_suffix(" step"))
                .and_then(|n| n.parse::<usize>().ok())
        {
            folded += n;
        }
    }
    (folded, steps)
}

/// Lays out steps after folded earlier ones, the answer, and the footers, within limit.
fn layout(folded: usize, steps: &[String], resp: &ChatResponse, limit: usize) -> Vec<String> {
    let limit = clamp(limit);
    let body = body(resp);
    let shown = if folded > 0 {
        compose_lines(&count_line(folded), &bullets(steps))
    } else {
        bullets(steps)
    };
    let full = compose(&[&shown, &body, &footers(resp, None)]);
    if full.len() <= limit {
        return vec![full];
    }
    let count = count_line(folded + steps.len());
    let collapsed = compose(&[&count, &body, &footers(resp, None)]);
    if collapsed.len() <= limit {
        return vec![collapsed];
    }
    let trimmed = compose(&[&count, &body, &footers(resp, Some(FOOTER_KEEP))]);
    chunk(&trimmed, limit)
}

/// One step as a small grey bullet.
fn bullet(step: &str) -> String {
    format!("{BULLET}{step}")
}

/// Every step as a bullet, one per line.
fn bullets(steps: &[String]) -> String {
    steps
        .iter()
        .map(|s| bullet(s))
        .collect::<Vec<_>>()
        .join("\n")
}

/// n steps folded into one subtext count. Empty for none.
fn count_line(n: usize) -> String {
    if n == 0 {
        String::new()
    } else {
        format!("-# {n} {}", plural(n))
    }
}

/// The word for n steps.
fn plural(n: usize) -> &'static str {
    if n == 1 { "step" } else { "steps" }
}

/// The answer text, or what the status means when there is none, plus the approval line.
fn body(resp: &ChatResponse) -> String {
    let mut body = resp.text.trim().to_owned();
    if body.is_empty() && resp.confirmation.is_none() {
        body = match resp.status.as_str() {
            "step_limit" => "I could not finish within the allowed number of steps.".into(),
            "stalled" => "I kept repeating myself without getting further; try rephrasing.".into(),
            "deadline" => "That took too long; please try again.".into(),
            "cancelled" => "Cancelled.".into(),
            _ => "I have no answer for that.".into(),
        };
    }
    if let Some(c) = &resp.confirmation {
        if !body.is_empty() {
            body.push_str("\n\n");
        }
        let _ = write!(body, "**`{}` needs your approval:** {}", c.tool, c.summary);
    }
    body
}

/// Sources, Tools, and Memory used, each only when it has content. keep caps each list.
fn footers(resp: &ChatResponse, keep: Option<usize>) -> String {
    let mut sections = Vec::new();
    if !resp.citations.is_empty() {
        let mut s = String::from("**Sources**");
        for (i, c) in kept(&resp.citations, keep).iter().enumerate() {
            let _ = write!(s, "\n{}. {c}", i + 1);
        }
        if let Some(n) = hidden(resp.citations.len(), keep) {
            let _ = write!(s, "\nand {n} more");
        }
        sections.push(s);
    }
    if !resp.tools.is_empty() {
        let names: Vec<String> = kept(&resp.tools, keep)
            .iter()
            .map(|t| {
                if t.ok {
                    format!("`{}`", t.tool)
                } else {
                    format!("`{}` (failed)", t.tool)
                }
            })
            .collect();
        let mut s = format!("**Tools** {}", names.join(", "));
        if let Some(n) = hidden(resp.tools.len(), keep) {
            let _ = write!(s, ", and {n} more");
        }
        sections.push(s);
    }
    if !resp.memories.is_empty() {
        let mut s = String::from("**Memory used**");
        for m in kept(&resp.memories, keep) {
            let _ = write!(s, "\n- {m}");
        }
        if let Some(n) = hidden(resp.memories.len(), keep) {
            let _ = write!(s, "\nand {n} more");
        }
        sections.push(s);
    }
    sections.join("\n")
}

/// The first keep items, or all of them.
fn kept<T>(items: &[T], keep: Option<usize>) -> &[T] {
    keep.and_then(|k| items.get(..k)).unwrap_or(items)
}

/// How many items keep leaves out, if any.
fn hidden(total: usize, keep: Option<usize>) -> Option<usize> {
    keep.map(|k| total.saturating_sub(k)).filter(|&n| n > 0)
}

/// Non-empty parts separated by a blank line.
fn compose(parts: &[&str]) -> String {
    parts
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Two non-empty parts on consecutive lines.
fn compose_lines(first: &str, second: &str) -> String {
    [first, second]
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The first sendable piece of text.
fn fit(text: String, limit: usize) -> String {
    if text.len() <= limit {
        return text;
    }
    chunk(&text, limit).into_iter().next().unwrap_or_default()
}

/// A configured limit held to what Discord accepts.
fn clamp(limit: usize) -> usize {
    limit.clamp(1, MAX_MESSAGE)
}
