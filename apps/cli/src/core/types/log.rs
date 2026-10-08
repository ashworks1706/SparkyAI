//! Captured output lines.

use chrono::{DateTime, Local};

/// Which output a log line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    /// Child stdout.
    Out,
    /// Child stderr.
    Err,
    /// A note from the console about the unit.
    Meta,
}

/// One captured line.
#[derive(Debug, Clone)]
pub struct LogLine {
    /// When it was captured.
    pub at: DateTime<Local>,
    /// Source.
    pub stream: Stream,
    /// Content without the trailing newline.
    pub text: String,
}

impl LogLine {
    /// A line captured now.
    pub fn now(stream: Stream, text: impl Into<String>) -> Self {
        Self {
            at: Local::now(),
            stream,
            text: text.into(),
        }
    }
}
