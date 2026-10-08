//! Conversation scope: who reads a turn, what it carries, and ending one.

use serde::{Deserialize, Serialize};

/// Who may read a turn. Public turns get no personal memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    /// Everyone in the channel reads it.
    Public,
    /// Only the asker reads it.
    Private,
}

/// Media types a model is sent. Anything else is dropped.
pub const IMAGE_MEDIA_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/gif", "image/webp"];

/// One image attached to a message. Mirrors engine::core::types::conversation::image::Attachment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Attachment {
    /// Direct link to the image.
    pub url: String,
    /// Media type Discord reported, one of IMAGE_MEDIA_TYPES.
    pub media_type: String,
}

impl Attachment {
    /// An attachment, or None when the media type is not one a model is sent.
    pub fn new(url: impl Into<String>, media_type: impl Into<String>) -> Option<Self> {
        let kind = media_type
            .into()
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if !IMAGE_MEDIA_TYPES.contains(&kind.as_str()) {
            return None;
        }
        let url = url.into();
        if url.is_empty() {
            return None;
        }
        Some(Self {
            url,
            media_type: kind,
        })
    }
}

/// One non-image file attached to a message. Mirrors engine::core::types::conversation::file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FileAttachment {
    /// Direct link to the file.
    pub url: String,
    /// File name as uploaded.
    pub name: String,
    /// Media type Discord reported, empty when it reported none.
    pub media_type: String,
    /// Size in bytes Discord reported.
    pub size: u64,
}

/// Body of POST /conversation/reset, ending the open conversations of a user in one channel.
#[derive(Debug, Serialize)]
pub struct ResetRequest {
    /// Discord user id.
    #[serde(rename = "user_id")]
    pub user: String,
    /// Guild id.
    #[serde(rename = "tenant_id")]
    pub tenant: String,
    /// Channel or thread id.
    #[serde(rename = "channel_id")]
    pub channel: String,
}

/// Reply to POST /conversation/reset.
#[derive(Debug, Deserialize)]
pub struct ResetResponse {
    /// Conversations ended.
    pub ended: u64,
}
