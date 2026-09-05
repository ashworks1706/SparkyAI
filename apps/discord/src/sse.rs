//! Server-sent event framing, for the engine's `/chat/stream`.

/// Pulls every complete `event:`/`data:` frame out of `buf`, leaving any partial tail behind.
///
/// Returns `(event name, data)` pairs in arrival order. A frame with no `event:` line reports an
/// empty name, which is what the SSE default (`message`) means here.
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
