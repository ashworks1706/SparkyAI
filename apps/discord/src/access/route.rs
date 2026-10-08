//! Where a turn is anchored and the request it produces.

use serenity::all::{ChannelId, ChannelType, GuildId, MessageFlags, UserId};

use crate::core::types::{Attachment, ChatRequest, Visibility};

/// Why a message is one the bot answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// A direct message. Every one is a turn, and no thread exists to open.
    Direct,
    /// A message in a channel that addresses the bot. The answer opens a thread.
    Opening,
    /// A reply in a thread to something the bot said there.
    Reply,
}

/// The kind of place a message arrived in. The three are exclusive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Arrived {
    /// A direct message: no guild, and no thread to open.
    Direct,
    /// A thread, which is where a conversation lives.
    Thread,
    /// A guild channel outside any thread.
    #[default]
    Channel,
}

/// Where one message arrived and how it addresses the bot.
#[derive(Debug, Clone, Copy, Default)]
pub struct Arrival {
    /// The kind of place it arrived in.
    pub at: Arrived,
    /// It mentions the bot.
    pub mentions_bot: bool,
    /// It replies to something the bot said.
    pub replies_to_bot: bool,
}

/// Why the bot answers this message, or None when the message is not for it.
pub fn trigger(arrival: Arrival) -> Option<Trigger> {
    match arrival.at {
        Arrived::Direct => Some(Trigger::Direct),
        Arrived::Thread => arrival.replies_to_bot.then_some(Trigger::Reply),
        Arrived::Channel => {
            (arrival.mentions_bot || arrival.replies_to_bot).then_some(Trigger::Opening)
        }
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
    reply_to: Option<String>,
    images: Vec<Attachment>,
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
        reply_to,
        images,
        files: Vec::new(),
    }
}
