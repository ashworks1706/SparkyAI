//! The one message a turn lives in: steps while it runs, then the answer and its footers.

use std::fmt::Write;
use std::time::{Duration, Instant};

use crate::core::types::{ChatResponse, Progress};
use crate::render::reply::{MAX_MESSAGE, chunk};

/// Header of the message while a turn runs, after the spinner frame.
pub const THINKING: &str = "**Sparky is working on it…**";

/// Header of the message once the answer is being written.
pub const ANSWERING: &str = "**Sparky is answering…**";

/// Frames the header spinner cycles through, one per edit of the running card.
const SPINNER: [&str; 4] = ["\u{25d0}", "\u{25d3}", "\u{25d1}", "\u{25d2}"];

/// Prefix of one step line. The emoji the engine sends is the only marker a step carries.
const BULLET: &str = "-# ";

/// Items each footer keeps once footers are trimmed to fit.
const FOOTER_KEEP: usize = 3;

/// Longest label on a source button. Discord allows 80.
const LABEL_CHARS: usize = 40;

/// One step line and the key of the line it replaces, when it replaces one.
#[derive(Debug, Clone)]
struct Step {
    slot: Option<String>,
    text: String,
}

/// The steps the engine reported for one turn, in order, and the answer written so far.
#[derive(Debug, Default, Clone)]
pub struct Steps {
    lines: Vec<Step>,
    draft: Option<String>,
}

impl Steps {
    /// Applies one progress update and says whether what the card shows changed.
    pub fn apply(&mut self, progress: &Progress) -> bool {
        if progress.draft {
            if progress.clear {
                return self.draft.take().is_some();
            }
            let text = progress.text.trim();
            if text.is_empty() || self.draft.as_deref() == Some(text) {
                return false;
            }
            self.draft = Some(text.to_owned());
            return true;
        }
        match (progress.clear, progress.slot.as_deref()) {
            (true, Some(slot)) => self.clear(slot),
            (_, slot) => self.push(slot, &progress.text),
        }
    }

    /// The answer written so far, when there is one.
    pub fn draft(&self) -> Option<&str> {
        self.draft.as_deref()
    }

    /// Records a step and says whether the list changed. A slot writes over the line of that slot.
    pub fn push(&mut self, slot: Option<&str>, text: &str) -> bool {
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        if let Some(key) = slot
            && let Some(held) = self
                .lines
                .iter_mut()
                .find(|line| line.slot.as_deref() == Some(key))
        {
            if held.text == text {
                return false;
            }
            text.clone_into(&mut held.text);
            return true;
        }
        if self.lines.last().is_some_and(|last| last.text == text) {
            return false;
        }
        self.lines.push(Step {
            slot: slot.map(str::to_owned),
            text: text.to_owned(),
        });
        true
    }

    /// Takes back the line of slot, and says whether one was there.
    pub fn clear(&mut self, slot: &str) -> bool {
        let before = self.lines.len();
        self.lines.retain(|line| line.slot.as_deref() != Some(slot));
        self.lines.len() != before
    }

    /// The steps so far, in order.
    pub fn lines(&self) -> Vec<String> {
        self.lines.iter().map(|line| line.text.clone()).collect()
    }
}

/// Paces edits of the running card: at most one per interval.
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

/// The message while the turn runs: steps, then answer so far. frame advances spinner per edit.
pub fn thinking(steps: &[String], draft: Option<&str>, limit: usize, frame: usize) -> String {
    let limit = clamp(limit);
    let draft = draft.map(str::trim).filter(|d| !d.is_empty());
    let title = if draft.is_some() { ANSWERING } else { THINKING };
    let header = format!("{} {title}", SPINNER[frame % SPINNER.len()]);
    let mut top = header.clone();
    for hidden in 0..=steps.len() {
        let mut out = header.clone();
        if hidden > 0 {
            let _ = write!(out, "\n-# {hidden} earlier {}", plural(hidden));
        }
        for step in steps.iter().skip(hidden) {
            out.push('\n');
            out.push_str(&bullet(step));
        }
        let full = match draft {
            Some(text) => format!("{out}\n\n{text}"),
            None => out.clone(),
        };
        if full.len() <= limit {
            return full;
        }
        top = out;
    }
    let Some(text) = draft else {
        return fit(header, limit);
    };
    // Every step is folded and the draft still overflows. Keeps its newest part.
    let room = limit.saturating_sub(top.len() + "\n\n\u{2026}".len());
    format!("{top}\n\n\u{2026}{}", newest(text, room).trim_start())
}

/// The last bytes of text that fit in room, starting at a character boundary.
fn newest(text: &str, room: usize) -> &str {
    let mut start = text.len().saturating_sub(room);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

/// The finished turn: steps, answer, then the footers that have content.
pub fn answer(steps: &[String], resp: &ChatResponse, limit: usize) -> Vec<String> {
    layout(0, steps, resp, limit)
}

/// The card after an approval: old card's steps, a step for the decision, then the resumed answer.
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
        let Some(rest) = line.strip_prefix(BULLET) else {
            continue;
        };
        // A folded step count shares the bullet prefix and is matched first.
        if let Some(n) = rest
            .strip_suffix(" steps")
            .or_else(|| rest.strip_suffix(" step"))
            .and_then(|n| n.parse::<usize>().ok())
        {
            folded += n;
        } else {
            steps.push(rest.to_owned());
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

/// One step as a small grey line.
fn bullet(step: &str) -> String {
    format!("{BULLET}{step}")
}

/// Every step as a line of its own.
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

/// What sits under the answer: unlinked sources and memory the answer drew on. keep caps each list.
fn footers(resp: &ChatResponse, keep: Option<usize>) -> String {
    let mut sections = Vec::new();
    let unlinked: Vec<&str> = resp
        .citations
        .iter()
        .filter(|c| c.url.as_deref().unwrap_or("").trim().is_empty())
        .map(|c| c.title.as_str())
        .collect();
    if !unlinked.is_empty() {
        let shown = kept(&unlinked, keep).join(", ");
        // The subtext prefix is reserved for step lines read back out of the card.
        let mut s = format!("**\u{1f4da} Also from** {shown}");
        if let Some(n) = hidden(unlinked.len(), keep) {
            let _ = write!(s, ", and {n} more");
        }
        sections.push(s);
    }
    if !resp.memories.is_empty() {
        let mut s = String::from("**\u{1f9e0} Memory used**");
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

/// The label a source button carries: its title, held to what Discord shows.
pub fn source_label(title: &str) -> String {
    let title = title.replace('_', " ");
    let title = title.trim();
    if title.chars().count() <= LABEL_CHARS {
        return title.to_owned();
    }
    let kept: String = title.chars().take(LABEL_CHARS - 1).collect();
    format!("{}\u{2026}", kept.trim_end())
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
