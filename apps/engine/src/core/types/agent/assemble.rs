//! Budget, Sections, and Assembled are the inputs and output of context assembly.

use crate::core::types::conversation::message::Message;
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::memory::Memory;

/// Budgets for one assembled prompt, in estimated tokens. Default is implemented in
/// core::config.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Whole prompt, everything included.
    pub total: usize,
    /// Cap on the evidence section.
    pub evidence: usize,
    /// Cap on prior turns.
    pub history: usize,
    /// Cap on the memory section.
    pub memory: usize,
    /// Cap on the capabilities section.
    pub capabilities: usize,
    /// Characters per token the estimator assumes. Never zero.
    pub chars_per_token: usize,
}

/// The text assembly writes around the sections. Every field is configurable.
#[derive(Debug, Clone, Copy)]
pub struct Templates<'a> {
    /// Line naming the user, with {user} and {roles}.
    pub role_line: &'a str,
    /// Line naming a user who holds no roles, with {user}.
    pub role_line_no_roles: &'a str,
    /// Heading above recalled memories.
    pub memory_header: &'a str,
    /// Heading above retrieved evidence.
    pub evidence_header: &'a str,
    /// Line written when retrieval found nothing.
    pub no_evidence_line: &'a str,
    /// Heading above what the model may do.
    pub capabilities_header: &'a str,
    /// Line naming the current date, with {date}.
    pub date_line: &'a str,
}

/// Default line naming the user, with {user} and {roles}.
pub const ROLE_LINE: &str = "The user is `{user}`. Roles: {roles}.";
/// Default line naming a user who holds no roles, with {user}.
pub const ROLE_LINE_NO_ROLES: &str = "The user is `{user}`. They hold no special roles.";
/// Default heading above recalled memories.
pub const MEMORY_HEADER: &str = "What you remember about this user:";
/// Default line naming the current date, with {date}.
pub const DATE_LINE: &str = "Today is {date}. Evidence rows are labelled by day or date; read \
                             the label the question asks for, never the first value in a row.";
/// Default heading above retrieved evidence.
pub const EVIDENCE_HEADER: &str = "Knowledge base results for this question, closest match \
                                   first. These were retrieved for you before you were called. \
                                   Answer from them and from tool output only, and cite the \
                                   bracketed number of every entry you use.";
/// Default line written when retrieval found nothing.
pub const NO_EVIDENCE_LINE: &str = "The knowledge base returned nothing for this question. \
                                    Call the search_ tool for the topic before you answer, or \
                                    say you do not have it.";
/// Default line added to a call that is offered no tools.
pub const ANSWER_ONLY_LINE: &str = "You have no tools on this step. Answer the user now in plain \
                                    text from what you already have, or say what you could not \
                                    find. Do not write a tool call.";
/// Default heading above what the model may do.
pub const CAPABILITIES_HEADER: &str = "What you can do. Each line is a name, how it runs, and \
                                       what it does.";

impl Default for Templates<'_> {
    fn default() -> Self {
        Self {
            role_line: ROLE_LINE,
            role_line_no_roles: ROLE_LINE_NO_ROLES,
            memory_header: MEMORY_HEADER,
            evidence_header: EVIDENCE_HEADER,
            no_evidence_line: NO_EVIDENCE_LINE,
            capabilities_header: CAPABILITIES_HEADER,
            date_line: DATE_LINE,
        }
    }
}

/// Everything that can go into a prompt.
#[derive(Debug, Default)]
pub struct Sections<'a> {
    /// Versioned system instructions.
    pub system: &'a str,
    /// Recalled memories, best first.
    pub memory: &'a [Memory],
    /// Retrieved evidence, best first.
    pub evidence: &'a [Evidence],
    /// Prior turns, oldest first.
    pub history: &'a [Message],
    /// The messages of this request after the input: tool calls and their results, oldest
    /// first. Always kept, after the input.
    pub turn: &'a [Message],
    /// What the model may do, rendered. Empty writes no section.
    pub capabilities: &'a str,
    /// The current message from the user.
    pub input: &'a str,
    /// Today's date as the model should read it. Empty writes no date line.
    pub date: &'a str,
    /// Wording written around the sections.
    pub templates: Templates<'a>,
}

/// The assembled prompt plus what was left out.
#[derive(Debug)]
pub struct Assembled {
    /// Messages to send, in order.
    pub messages: Vec<Message>,
    /// Rough token estimate of the whole thing.
    pub estimated_tokens: usize,
    /// Evidence chunks that made it in.
    pub evidence_used: usize,
    /// Memories that made it in, counted from the first.
    pub memory_used: usize,
}
