//! Per-unit log files and the bounded log buffer with search.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use crate::core::types::{LogLine, Stream};

/// Persists console output in one file per unit.
#[derive(Debug)]
pub struct LogWriter {
    dir: PathBuf,
    files: HashMap<String, BufWriter<File>>,
}

impl LogWriter {
    /// Creates the log directory if it does not exist.
    pub fn new(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            files: HashMap::new(),
        })
    }

    /// Appends and flushes one line to the unit log file.
    pub fn append(&mut self, unit: &str, line: &LogLine) -> std::io::Result<()> {
        let file = match self.files.entry(log_name(unit)) {
            Entry::Occupied(open) => open.into_mut(),
            Entry::Vacant(slot) => {
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(self.dir.join(slot.key()))?;
                slot.insert(BufWriter::new(file))
            }
        };
        let stream = match line.stream {
            Stream::Out => "out",
            Stream::Err => "err",
            Stream::Meta => "meta",
        };
        writeln!(
            file,
            "{} {stream} {}",
            line.at.format("%Y-%m-%dT%H:%M:%S%:z"),
            line.text
        )?;
        file.flush()
    }
}

fn log_name(unit: &str) -> String {
    let stem: String = unit
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
                c
            } else {
                '-'
            }
        })
        .collect();
    format!("{}.log", stem.trim_matches('-'))
}

/// Keeps the newest cap lines of one unit.
#[derive(Debug)]
pub struct LogBuffer {
    lines: VecDeque<LogLine>,
    cap: usize,
}

impl LogBuffer {
    /// An empty buffer that holds at most cap lines.
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

    /// Index of the next line containing needle, case-insensitive, after from, wrapping
    /// around. Setting backwards searches toward older lines.
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
