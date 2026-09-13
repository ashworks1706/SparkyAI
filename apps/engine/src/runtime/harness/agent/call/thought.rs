//! What the model was thinking on one step, and the answer with that thinking taken out.

/// Opening tag of inline reasoning.
const OPEN: &str = "<think>";
/// Closing tag of inline reasoning.
const CLOSE: &str = "</think>";

/// The thought a response carries and the text left for the user.
///
/// Reasoning the provider returned in its own field takes precedence. Inline reasoning between
/// think tags is removed from the text, including an unclosed trailing block.
pub fn split(reasoning: &str, content: &str) -> (Option<String>, String) {
    let (inline, visible) = lift(content);
    let thought = [reasoning.trim(), inline.trim()]
        .into_iter()
        .find(|text| !text.is_empty())
        .map(str::to_owned);
    (thought, visible)
}

/// The inline reasoning of content and what is left once it is removed.
fn lift(content: &str) -> (String, String) {
    let mut thought = String::new();
    let mut visible = String::new();
    let mut rest = content;
    while let Some(open) = rest.find(OPEN) {
        visible.push_str(&rest[..open]);
        let after = &rest[open + OPEN.len()..];
        let Some(close) = after.find(CLOSE) else {
            // An unclosed tag runs to the end of the content.
            push_line(&mut thought, after);
            return (thought, visible.trim().to_owned());
        };
        push_line(&mut thought, &after[..close]);
        rest = &after[close + CLOSE.len()..];
    }
    visible.push_str(rest);
    (thought, visible.trim().to_owned())
}

/// Appends a block to the thought, on its own line.
fn push_line(thought: &mut String, block: &str) {
    let block = block.trim();
    if block.is_empty() {
        return;
    }
    if !thought.is_empty() {
        thought.push('\n');
    }
    thought.push_str(block);
}
