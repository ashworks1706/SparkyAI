//! Frames of /chat/stream and the updates they decode to.

use serde::Deserialize;

use super::chat::ChatResponse;
use super::error::EngineError;

/// The engine error frame on /chat/stream.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorFrame {
    /// What went wrong.
    pub error: String,
    /// The status the JSON route would have returned.
    #[serde(default)]
    pub status: Option<u16>,
}

/// One line of progress from /chat/stream. slot names the line this one writes over.
#[derive(Debug, Clone, Deserialize)]
pub struct Progress {
    /// Ready-to-display sentence.
    pub text: String,
    /// The line this replaces, when it replaces one.
    #[serde(default)]
    pub slot: Option<String>,
    /// Removes the line of slot instead of writing text there.
    #[serde(default)]
    pub clear: bool,
    /// The text is the answer written so far, shown as the body.
    #[serde(default)]
    pub draft: bool,
}

/// What arrives while a streamed turn runs.
#[derive(Debug)]
pub enum Update {
    /// Something happened worth showing.
    Progress(Progress),
    /// The turn finished.
    Answer(Box<ChatResponse>),
    /// The turn failed.
    Failed(EngineError),
}
