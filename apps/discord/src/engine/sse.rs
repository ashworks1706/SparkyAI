//! Server-sent event framing for the engine /chat/stream endpoint.

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
