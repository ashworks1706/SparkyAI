//! Latest posts on the official ASU and Sun Devil Athletics channels.

use super::{LiveSource, Param};

/// The official ASU YouTube channels.
pub struct SocialMedia;

const ACCOUNTS: &[&str] = &["asu", "sun devil athletics"];

const PARAMS: &[Param] = &[
    Param::one_of("account", "Account. Leave out for every account.", ACCOUNTS),
    Param::text("keywords", "Words in the post.").example("football"),
];

impl LiveSource for SocialMedia {
    fn key(&self) -> &'static str {
        "social_media"
    }

    fn hint(&self) -> &'static str {
        "official ASU video posts"
    }

    fn category(&self) -> &'static str {
        "social"
    }

    fn params(&self) -> &'static [Param] {
        PARAMS
    }
}
