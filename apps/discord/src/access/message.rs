//! What a message carries: its question text, attachments, quote, and whether it addresses the bot.

use serenity::all::{RoleId, UserId};

use crate::core::types::{Attachment, FileAttachment};

/// Longest thread name Discord accepts, in characters.
pub const THREAD_NAME_MAX: usize = 100;

/// The message content with every mention of the bot and its managed role removed, trimmed.
pub fn strip_mentions(content: &str, bot: UserId, role: Option<RoleId>) -> String {
    let mut out = content
        .replace(&format!("<@{bot}>"), " ")
        .replace(&format!("<@!{bot}>"), " ");
    if let Some(role) = role {
        out = out.replace(&format!("<@&{role}>"), " ");
    }
    out.trim().to_owned()
}

/// The role Discord manages for the bot: the one whose tags name it as the bot.
pub fn bot_role(
    roles: impl IntoIterator<Item = (RoleId, Option<UserId>)>,
    bot: UserId,
) -> Option<RoleId> {
    roles
        .into_iter()
        .find_map(|(id, tagged)| (tagged == Some(bot)).then_some(id))
}

/// Whether a message addresses the bot, by its user or by its managed role.
pub fn addresses_bot(
    mentions_user: bool,
    mentioned_roles: &[RoleId],
    role: Option<RoleId>,
) -> bool {
    mentions_user || role.is_some_and(|r| mentioned_roles.contains(&r))
}

/// The text of the message a reply answers, when the bot wrote it and it holds text.
pub fn quoted(author: Option<UserId>, content: &str, me: UserId) -> Option<String> {
    if author != Some(me) {
        return None;
    }
    let text = content.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// The attachments with an image content type, at most most of them.
pub fn images<'a>(
    attachments: impl IntoIterator<Item = (&'a str, Option<&'a str>)>,
    most: usize,
) -> Vec<Attachment> {
    attachments
        .into_iter()
        .filter_map(|(url, kind)| Attachment::new(url, kind.unwrap_or_default()))
        .take(most)
        .collect()
}

/// The files of a message that are not images, no larger than max_bytes, at most most of them.
pub fn files<'a>(
    attachments: impl IntoIterator<Item = (&'a str, &'a str, Option<&'a str>, u64)>,
    most: usize,
    max_bytes: u64,
) -> Vec<FileAttachment> {
    attachments
        .into_iter()
        .filter(|(url, _, kind, size)| {
            !url.is_empty()
                && *size <= max_bytes
                && Attachment::new(*url, kind.unwrap_or_default()).is_none()
        })
        .take(most)
        .map(|(url, name, kind, size)| FileAttachment {
            url: url.to_owned(),
            name: name.to_owned(),
            media_type: kind
                .unwrap_or_default()
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase(),
            size,
        })
        .collect()
}

/// A thread name for a question: one line, within the Discord limit, never empty.
pub fn thread_name(question: &str) -> String {
    let line = question.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.is_empty() {
        return "Question".into();
    }
    truncate(&line, THREAD_NAME_MAX)
}

/// Cuts text to at most max characters, ending in an ellipsis when cut.
fn truncate(text: &str, max: usize) -> String {
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
