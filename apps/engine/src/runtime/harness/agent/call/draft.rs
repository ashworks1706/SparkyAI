//! What a streaming call has written so far, released a block at a time as it grows.

use super::thought;
use crate::core::types::model::ModelDelta;

/// Opening tag of inline reasoning, which a partly written answer must not show.
const OPEN: &str = "<think>";

/// Something ready to show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Release {
    /// The reasoning so far, while the model is still reasoning.
    Reasoning(String),
    /// Everything the model reasoned, once the answer has begun.
    Thought(String),
    /// The answer so far, up to the end of a block.
    Answer(String),
}

/// The reasoning and answer text of one streaming call.
#[derive(Debug, Default)]
pub struct Draft {
    reasoning: String,
    text: String,
    block_chars: usize,
    shown_reasoning: usize,
    shown_answer: String,
    thought_released: bool,
    withheld: bool,
}

impl Draft {
    /// A draft that holds back at most block_chars of answer text waiting for a block to end.
    pub fn new(block_chars: usize) -> Self {
        Self {
            block_chars,
            ..Self::default()
        }
    }

    /// Adds a piece of the completion and returns what became ready to show, in order.
    pub fn push(&mut self, delta: ModelDelta) -> Vec<Release> {
        match delta {
            ModelDelta::Reasoning(piece) => self.reasoning.push_str(&piece),
            ModelDelta::Text(piece) => self.text.push_str(&piece),
        }
        let (inline, mut visible) = thought::split("", &self.text);
        // The split trims, so a trailing break is restored.
        if !visible.is_empty() && self.text.ends_with(char::is_whitespace) {
            visible.push(' ');
        }
        let reasoning = if self.reasoning.trim().is_empty() {
            inline.unwrap_or_default()
        } else {
            self.reasoning.clone()
        };
        let mut out = Vec::new();
        let answering = !visible.trim().is_empty() && !is_opening_tag(&visible);
        if answering {
            if !self.thought_released && !reasoning.trim().is_empty() {
                self.thought_released = true;
                out.push(Release::Thought(reasoning));
            }
            if let Some(answer) = self.next_answer(&visible) {
                out.push(Release::Answer(answer));
            }
        } else if reasoning.len() > self.shown_reasoning
            && ends_block(&reasoning, self.shown_reasoning)
        {
            self.shown_reasoning = reasoning.len();
            out.push(Release::Reasoning(reasoning));
        }
        out
    }

    /// Stops releasing answer text for the rest of the call.
    pub fn withhold(&mut self) {
        self.withheld = true;
    }

    /// Whether any answer text was released.
    pub fn shown(&self) -> bool {
        !self.shown_answer.is_empty()
    }

    /// The answer up to the end of its last block, when that is more than was shown before.
    fn next_answer(&mut self, visible: &str) -> Option<String> {
        if self.withheld {
            return None;
        }
        // A closed tag can shorten the visible text, which then restarts from the beginning.
        let from = if visible.starts_with(&self.shown_answer) {
            self.shown_answer.len()
        } else {
            0
        };
        let fresh = &visible[from..];
        let cut = block_end(fresh).or_else(|| {
            (fresh.chars().count() >= self.block_chars)
                .then(|| fresh.rfind(char::is_whitespace).unwrap_or(fresh.len()))
        })?;
        let answer = visible[..from + cut].trim_end().to_owned();
        if answer.is_empty() || answer == self.shown_answer {
            return None;
        }
        self.shown_answer.clone_from(&answer);
        Some(answer)
    }
}

/// Whether text from shown on holds the end of a sentence or a line.
fn ends_block(text: &str, shown: usize) -> bool {
    text.get(shown..)
        .is_some_and(|fresh| block_end(fresh).is_some())
}

/// The byte offset just past the last sentence end or line break in text.
fn block_end(text: &str) -> Option<usize> {
    let mut end = None;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let next_is_space = chars.peek().is_some_and(|(_, n)| n.is_whitespace());
        if c == '\n' || (matches!(c, '.' | '!' | '?' | ':') && next_is_space) {
            end = Some(i + c.len_utf8());
        }
    }
    end
}

/// Whether text is the start of an inline reasoning tag still being written.
fn is_opening_tag(text: &str) -> bool {
    text.len() < OPEN.len() && OPEN.starts_with(text)
}
