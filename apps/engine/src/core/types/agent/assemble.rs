//! Budget, Sections, and Assembled are the inputs and output of context assembly.

use chrono::{DateTime, Utc};

use crate::core::types::conversation::message::Message;
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::knowledge::route::Route;
use crate::core::types::memory::Memory;

/// Budgets for one assembled prompt, in estimated tokens. Default is implemented in core::config.
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
    /// Cap on the quoted message a reply answers.
    pub reply: usize,
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
    /// Line written when the router skipped retrieval as small talk.
    pub no_retrieval_line: &'a str,
    /// Line written when the router skipped retrieval because the answer has to be current.
    pub live_only_line: &'a str,
    /// Line written when the knowledge base could not be read.
    pub no_index_line: &'a str,
    /// Heading above what the model may do.
    pub capabilities_header: &'a str,
    /// Line naming the current date, with {date}.
    pub date_line: &'a str,
    /// Heading above the message of yours a reply answers.
    pub reply_header: &'a str,
    /// Line closing a tool result cut to fit the prompt, with {chars}.
    pub result_cut_line: &'a str,
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
pub const EVIDENCE_HEADER: &str = "Stored copies of pages from the knowledge base, closest match \
                                   first, each with when it was fetched. They may be out of date \
                                   or may not answer the question. When the question needs current \
                                   information, such as hours today, shuttles, events, news or \
                                   scores, call search_live first. When these entries do not \
                                   answer it, call search_knowledge with the subject named in \
                                   full. Otherwise answer from them and from tool output only, \
                                   and cite the bracketed number of every entry you use.";
/// Default line written when retrieval found nothing.
pub const NO_EVIDENCE_LINE: &str = "The knowledge base returned nothing for this question. \
                                    Before you answer, call search_knowledge with the subject of \
                                    the question written out in keywords, or call search_live \
                                    when the answer has to be current. Say you do not have it \
                                    only after one of them comes back empty.";
/// Default line written when the router skipped retrieval as small talk.
pub const NO_RETRIEVAL_LINE: &str = "The knowledge base was not searched for this message: it \
                                     asks for no ASU fact. Answer it directly and briefly, and \
                                     cite nothing.";
/// Default line written when the router skipped retrieval because the answer has to be current.
pub const LIVE_ONLY_LINE: &str = "The knowledge base was not searched for this question: the \
                                  answer has to be current, and a stored copy would be out of \
                                  date. Call search_live with the subject written out in \
                                  keywords, and answer from what it returns, or say you do not \
                                  have it.";
/// Default line written when the knowledge base could not be read.
pub const NO_INDEX_LINE: &str = "The knowledge base could not be read for this question, so \
                                 nothing from it is in this prompt. That is a fault on our side, \
                                 not an answer. Call search_live with the subject written out in \
                                 keywords, open the page yourself if what comes back is thin, and \
                                 answer from that. Never tell the user the knowledge base is down.";
/// Default line sent back when a tool failed and the sandbox was not tried.
pub const SANDBOX_RETRY_LINE: &str = "A tool you called this turn failed, and you have not \
                                      opened the page yourself. Do that now: call run_sandbox \
                                      and fetch the page the question is about, then answer from \
                                      what it prints. Tell the user you could not find something \
                                      only after that has been tried.";
/// Default line added to a call that is offered no tools.
pub const ANSWER_ONLY_LINE: &str = "You have no tools on this step. Answer the user now in plain \
                                    text from what you already have, or say what you could not \
                                    find. Do not write a tool call.";
/// Default heading above the message of yours a reply answers.
pub const REPLY_HEADER: &str = "The user replied to this earlier message of yours. It is what \
                                they are answering, so read it as the immediate context of what \
                                they say next.";
/// Default line closing a tool result cut to fit the prompt, with {chars}.
pub const RESULT_CUT_LINE: &str = "[{chars} more characters of this result were cut to fit. \
                                   Search again with a narrower query to read them.]";
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
            no_retrieval_line: NO_RETRIEVAL_LINE,
            live_only_line: LIVE_ONLY_LINE,
            no_index_line: NO_INDEX_LINE,
            capabilities_header: CAPABILITIES_HEADER,
            date_line: DATE_LINE,
            reply_header: REPLY_HEADER,
            result_cut_line: RESULT_CUT_LINE,
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
    /// What the router decided about retrieval for this question.
    pub route: Route,
    /// Prior turns, oldest first.
    pub history: &'a [Message],
    /// Messages after the input: tool calls and results, oldest first, always kept.
    pub turn: &'a [Message],
    /// What the model may do, rendered. Empty writes no section.
    pub capabilities: &'a str,
    /// The current message from the user.
    pub input: &'a str,
    /// Today's date as the model should read it. Empty writes no date line.
    pub date: &'a str,
    /// When the prompt is built. None writes no age on evidence entries.
    pub now: Option<DateTime<Utc>>,
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
