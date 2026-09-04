//! Bounded per-unit log buffer with search.

use std::collections::VecDeque;

use crate::core::types::LogLine;

/// Keeps the newest `cap` lines of one unit.
#[derive(Debug)]
pub struct LogBuffer {
    lines: VecDeque<LogLine>,
    cap: usize,
}

impl LogBuffer {
    /// An empty buffer that holds at most `cap` lines.
    pub fn new(cap: usize) -> Self {
        Self {
            lines: VecDeque::with_capacity(cap.min(1024)),
            cap: cap.max(1),
        }
    }

    /// Appends, dropping the oldest line when full.
    pub fn push(&mut self, line: LogLine) {
        if self.lines.len() == self.cap {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    /// Lines oldest first.
    pub fn lines(&self) -> impl ExactSizeIterator<Item = &LogLine> {
        self.lines.iter()
    }

    /// Number of lines held.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Whether nothing has been captured.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Drops everything.
    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Index of the next line containing `needle` (case-insensitive) after `from`, wrapping
    /// around; `backwards` searches toward older lines.
    pub fn find(&self, needle: &str, from: usize, backwards: bool) -> Option<usize> {
        if needle.is_empty() || self.lines.is_empty() {
            return None;
        }
        let needle = needle.to_lowercase();
        let n = self.lines.len();
        (1..=n)
            .map(|step| {
                if backwards {
                    (from + n - step % n) % n
                } else {
                    (from + step) % n
                }
            })
            .find(|&i| self.lines[i].text.to_lowercase().contains(&needle))
    }
}
