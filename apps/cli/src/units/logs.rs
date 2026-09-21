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
    files: HashMap<String, (BufWriter<File>, u64)>,
    max_bytes: u64,
}

impl LogWriter {
    /// Creates the log directory if it does not exist. A file past max_bytes is rotated to one
    /// .1 file beside it, replacing the previous one; 0 never rotates.
    pub fn new(dir: impl Into<PathBuf>, max_bytes: u64) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            files: HashMap::new(),
            max_bytes,
        })
    }

    /// Appends and flushes one line to the unit log file, rotating it first when it is full.
    pub fn append(&mut self, unit: &str, line: &LogLine) -> std::io::Result<()> {
        let name = log_name(unit);
        let path = self.dir.join(&name);
        if self.max_bytes > 0
            && self
                .files
                .get(&name)
                .is_some_and(|(_, size)| *size >= self.max_bytes)
        {
            self.files.remove(&name);
            let mut rotated = path.clone().into_os_string();
            rotated.push(".1");
            std::fs::rename(&path, rotated)?;
        }
        let (file, size) = match self.files.entry(name) {
            Entry::Occupied(open) => open.into_mut(),
            Entry::Vacant(slot) => {
                let file = OpenOptions::new().create(true).append(true).open(&path)?;
                let size = file.metadata().map_or(0, |m| m.len());
                slot.insert((BufWriter::new(file), size))
            }
        };
        let stream = match line.stream {
            Stream::Out => "out",
            Stream::Err => "err",
            Stream::Meta => "meta",
        };
        let written = format!(
            "{} {stream} {}\n",
            line.at.format("%Y-%m-%dT%H:%M:%S%:z"),
            line.text
        );
        file.write_all(written.as_bytes())?;
        *size += written.len() as u64;
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

    /// Index of the next line containing needle, case-insensitive and wrapping.
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
