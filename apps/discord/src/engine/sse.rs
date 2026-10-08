//! Server-sent event framing and decoding for the engine /chat/stream endpoint.

use crate::core::types::{
    ChatResponse, ERROR_BODY_CHARS, EngineError, ErrorFrame, Progress, Update,
};

/// The update one named frame carries, or None for a frame the bot does not act on.
pub fn decode(name: &str, data: &str) -> Option<Update> {
    match name {
        "progress" => match serde_json::from_str::<Progress>(data) {
            Ok(p) => Some(Update::Progress(p)),
            Err(e) => {
                tracing::warn!(error = %e, "unreadable progress frame");
                None
            }
        },
        "answer" => Some(match serde_json::from_str::<ChatResponse>(data) {
            Ok(answer) => Update::Answer(Box::new(answer)),
            Err(e) => Update::Failed(EngineError::Transport(format!("bad body: {e}"))),
        }),
        "error" => {
            // The frame carries the status the JSON route would have used.
            let (status, body) = match serde_json::from_str::<ErrorFrame>(data) {
                Ok(f) => (f.status.unwrap_or(502), f.error),
                Err(e) => {
                    tracing::warn!(error = %e, "unreadable error frame");
                    (502, data.chars().take(ERROR_BODY_CHARS).collect())
                }
            };
            Some(Update::Failed(EngineError::Status { status, body }))
        }
        _ => None,
    }
}

/// Takes the bytes of every complete frame out of pending as text, leaving a partial frame behind.
pub fn take_complete(pending: &mut Vec<u8>) -> String {
    let Some(end) = pending.windows(2).rposition(|pair| pair == b"\n\n") else {
        return String::new();
    };
    let complete: Vec<u8> = pending.drain(..end + 2).collect();
    String::from_utf8_lossy(&complete).into_owned()
}

/// Pulls every complete event and data frame out of buf, leaving any partial tail behind.
pub fn drain_frames(buf: &mut String) -> Vec<(String, String)> {
    let mut frames = Vec::new();
    while let Some(end) = buf.find("\n\n") {
        let frame = buf[..end].to_owned();
        buf.drain(..end + 2);
        let mut name = String::new();
        let mut data = String::new();
        for line in frame.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                name.clear();
                name.push_str(rest.trim());
            } else if let Some(rest) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(rest.trim_start());
            }
        }
        if !data.is_empty() {
            frames.push((name, data));
        }
    }
    frames
}
