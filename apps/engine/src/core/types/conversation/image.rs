//! Images attached to a user's message.

use serde::{Deserialize, Serialize};

/// Media types a model is sent. Anything else is dropped at the edge.
pub const IMAGE_MEDIA_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/gif", "image/webp"];

/// One image attached to a user's message.
///
/// The link is what reaches the model, not the bytes: the edge holds them and every
/// OpenAI-compatible server fetches a URL. A server with no route to the host sees no image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attachment {
    /// Direct link to the image.
    pub url: String,
    /// Media type the edge reported, one of IMAGE_MEDIA_TYPES.
    pub media_type: String,
}

impl Attachment {
    /// The attachments of a request that name a media type a model is sent, at most most of them.
    /// The edge filters too; this is the HTTP surface not trusting its caller.
    pub fn accepted(images: Vec<Self>, most: usize) -> Vec<Self> {
        images
            .into_iter()
            .filter_map(|image| Self::new(image.url, image.media_type))
            .take(most)
            .collect()
    }

    /// An attachment, or None when the media type is not one a model is sent.
    pub fn new(url: impl Into<String>, media_type: impl Into<String>) -> Option<Self> {
        let media_type = media_type.into();
        let kind = media_type
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
