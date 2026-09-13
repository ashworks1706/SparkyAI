//! Where a turn is anchored, the request it produces, and the text derived from a question.

use serenity::all::{ChannelId, ChannelType, GuildId, MessageFlags, UserId};

use crate::core::types::{ChatRequest, Visibility};

/// Longest thread name Discord accepts, in characters.
pub const THREAD_NAME_MAX: usize = 100;

/// Where an /ask answer goes, decided before anything is posted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AskMode {
    /// Ephemeral followups only the asker sees.
    Private,
    /// Followups in the thread the command ran in.
    InThread,
    /// A new public thread opened from the question.
    NewThread,
}

/// Picks the AskMode for the private option and the kind of the channel the command ran in.
pub fn ask_mode(private: bool, kind: Option<ChannelType>) -> AskMode {
    if private {
        AskMode::Private
    } else if kind.is_some_and(is_thread) {
        AskMode::InThread
    } else {
        AskMode::NewThread
    }
}

/// Whether a channel kind is a thread.
pub fn is_thread(kind: ChannelType) -> bool {
    matches!(
        kind,
        ChannelType::PublicThread | ChannelType::PrivateThread | ChannelType::NewsThread
    )
}

/// Where a question was asked, as analytics names it: dm, thread, or channel.
pub fn place_name(dm: bool, in_thread: bool) -> &'static str {
    if dm {
        "dm"
    } else if in_thread {
        "thread"
    } else {
        "channel"
    }
}

/// The parent channel that counts for the allowlist: a thread parent, never a category.
pub fn thread_parent(kind: Option<ChannelType>, parent: Option<ChannelId>) -> Option<ChannelId> {
    if kind.is_some_and(is_thread) {
        parent
    } else {
        None
    }
}

/// Whether allow admits channel, or the thread parent of it. An empty allow admits all.
pub fn serves(allow: &[ChannelId], channel: ChannelId, parent: Option<ChannelId>) -> bool {
    allow.is_empty() || allow.contains(&channel) || parent.is_some_and(|p| allow.contains(&p))
}

/// Visibility and ephemeral-result flag for a confirmation, derived from the message's flags.
pub fn press_visibility(flags: Option<MessageFlags>) -> (Visibility, bool) {
    if flags.is_some_and(|f| f.contains(MessageFlags::EPHEMERAL)) {
        (Visibility::Private, true)
    } else {
        (Visibility::Public, false)
    }
}

/// The channel a turn is anchored to, and how the engine treats it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// Only the asker reads it. Personal memory applies.
    Private(ChannelId),
    /// A thread everyone reads. The engine continues the asker conversation there.
    Thread(ChannelId),
    /// A channel answered inline, one turn at a time.
    Inline(ChannelId),
}

/// The engine request for one question asked at place.
pub fn chat_request(
    place: Place,
    user: UserId,
    tenant: GuildId,
    roles: Vec<String>,
    message: String,
) -> ChatRequest {
    let (channel, visibility, continue_channel) = match place {
        Place::Private(c) => (c, Visibility::Private, true),
        Place::Thread(c) => (c, Visibility::Public, true),
        Place::Inline(c) => (c, Visibility::Public, false),
    };
    ChatRequest {
        user_id: user.to_string(),
        tenant_id: tenant.to_string(),
        channel_id: channel.to_string(),
        roles,
        conversation_id: None,
        message,
        visibility,
        continue_channel,
    }
}

/// Cuts text to at most max characters, ending in an ellipsis when cut.
pub fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    if max < 3 {
        return text.chars().take(max).collect();
    }
    let mut out = text
        .chars()
        .take(max - 3)
        .collect::<String>()
        .trim_end()
        .to_owned();
    out.push_str("...");
    out
}

/// A thread name for a question: one line, within the Discord limit, never empty.
pub fn thread_name(question: &str) -> String {
    let line = question.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.is_empty() {
        return "Question".into();
    }
    truncate(&line, THREAD_NAME_MAX)
}

/// The line that shows who asked what, within limit characters.
pub fn asked_header(name: &str, question: &str, limit: usize) -> String {
    let prefix = format!("**{name} asked:** ");
    let room = limit.saturating_sub(prefix.chars().count());
    format!("{prefix}{}", truncate(question.trim(), room))
}

/// The message content with every mention of the bot removed, trimmed.
pub fn strip_mentions(content: &str, bot: UserId) -> String {
    content
        .replace(&format!("<@{bot}>"), " ")
        .replace(&format!("<@!{bot}>"), " ")
        .trim()
        .to_owned()
}
