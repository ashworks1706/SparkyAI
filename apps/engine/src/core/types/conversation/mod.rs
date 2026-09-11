//! Visibility, ResetRequest, ResetResponse: who can read a conversation, and how one ends.

pub mod message;

use serde::{Deserialize, Serialize};

/// Who can read the exchange a request belongs to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    /// Others can read it, as in a guild thread.
    #[default]
    Public,
    /// Only the caller reads it, as in an ephemeral exchange or a one-to-one client.
    Private,
}

impl Visibility {
    /// Database column value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Private => "private",
        }
    }
}

/// POST /conversation/reset: ends every open conversation of the caller in one channel.
#[derive(Debug, Deserialize)]
pub struct ResetRequest {
    /// Caller id as the edge knows it.
    #[serde(rename = "user_id")]
    pub user: String,
    /// Tenant scope; defaults to the configured guild.
    #[serde(default, rename = "tenant_id")]
    pub tenant: Option<String>,
    /// Channel whose conversations end.
    #[serde(rename = "channel_id")]
    pub channel: String,
}

/// How many conversations ended.
#[derive(Debug, Serialize, Deserialize)]
pub struct ResetResponse {
    /// Conversations ended.
    pub ended: u64,
}
