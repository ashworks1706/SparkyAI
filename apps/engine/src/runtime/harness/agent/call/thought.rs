//! What the model was thinking on one step, and the answer with that thinking taken out.

/// Opening tag of inline reasoning.
const OPEN: &str = "<think>";
/// Closing tag of inline reasoning.
const CLOSE: &str = "</think>";

/// The thought a response carries and the text left for the user.
///
/// Reasoning the provider returned in a field of its own is believed first. A model that
/// writes its thinking inline between think tags has it lifted out of the answer, including
/// when the closing tag never arrived.
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
            // The step ran out of room mid-thought. What there is of it is still a thought.
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
