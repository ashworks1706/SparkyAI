//! Wire types mirrored from the engine HTTP contract, plus the client error.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Who may read a turn. Public turns get no personal memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    /// Everyone in the channel reads it.
    Public,
    /// Only the asker reads it.
    Private,
}

/// What the bot sends. Mirrors engine::core::types::chat::ChatRequest.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    /// Discord user id.
    pub user_id: String,
    /// Guild id.
    pub tenant_id: String,
    /// Channel or thread id the turn is anchored to.
    pub channel_id: String,
    /// Role names the member holds.
    pub roles: Vec<String>,
    /// Continue this conversation, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<Uuid>,
    /// The question.
    pub message: String,
    /// Who may read the turn.
    pub visibility: Visibility,
    /// Continue the latest open conversation of this user in channel_id with this visibility.
    pub continue_channel: bool,
    /// The bot message this one replies to, when it replies to one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<String>,
}

/// An action the engine is holding until the caller who asked answers it.
#[derive(Debug, Clone, Deserialize)]
pub struct Confirmation {
    /// Single-use token the buttons echo back to identify the held action.
    pub token: Uuid,
    /// Tool that would have run.
    pub tool: String,
    /// What would happen.
    pub summary: String,
}

/// What the engine returns. Mirrors engine::core::types::chat::ChatResponse.
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    /// Trace id.
    pub request_id: Uuid,
    /// Conversation to continue with.
    pub conversation_id: Uuid,
    /// The answer.
    pub text: String,
    /// Sources the answer rests on, best first.
    #[serde(default)]
    pub citations: Vec<Citation>,
    /// Set when the engine stopped to ask.
    #[serde(default)]
    pub confirmation: Option<Confirmation>,
    /// How the run ended.
    pub status: String,
    /// Remembered things the answer drew on, one line each.
    #[serde(default)]
    pub memories: Vec<String>,
}

/// One source under an answer. Mirrors engine::core::types::knowledge::evidence::Citation.
#[derive(Debug, Clone, Deserialize)]
pub struct Citation {
    /// Human-readable source name.
    pub title: String,
    /// Canonical page URL, when the source has one.
    #[serde(default)]
    pub url: Option<String>,
}

/// Engine call failures.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Could not reach the engine.
    #[error("engine unreachable: {0}")]
    Transport(String),
    /// The engine answered with an error status.
    #[error("engine returned {status}: {body}")]
    Status {
        /// HTTP status.
        status: u16,
        /// Body, truncated.
        body: String,
    },
}

/// The engine error frame on /chat/stream.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorFrame {
    /// What went wrong.
    pub error: String,
    /// The status the JSON route would have returned.
    #[serde(default)]
    pub status: Option<u16>,
}

/// One line of progress from /chat/stream. slot names the line this one writes over.
#[derive(Debug, Clone, Deserialize)]
pub struct Progress {
    /// Ready-to-display sentence.
    pub text: String,
    /// The line this replaces, when it replaces one.
    #[serde(default)]
    pub slot: Option<String>,
    /// Removes the line of slot instead of writing text there.
    #[serde(default)]
    pub clear: bool,
    /// The text is the answer written so far, shown as the body.
    #[serde(default)]
    pub draft: bool,
}

/// What arrives while a streamed turn runs.
#[derive(Debug)]
pub enum Update {
    /// Something happened worth showing.
    Progress(Progress),
    /// The turn finished.
    Answer(Box<ChatResponse>),
    /// The turn failed.
    Failed(EngineError),
}

/// Body of POST /confirm, answering an action the engine is holding.
#[derive(Debug, Serialize)]
pub struct ConfirmRequest {
    /// The token from the confirmation.
    pub token: Uuid,
    /// Whether to run it.
    pub approve: bool,
    /// Who is answering. The engine only accepts the caller who was asked.
    pub user_id: String,
    /// Guild scope.
    pub tenant_id: String,
    /// The conversation the held action belongs to.
    pub conversation_id: Uuid,
    /// Who may read the turn the action belongs to.
    pub visibility: Visibility,
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

/// Body of POST /profile/list.
#[derive(Debug, Serialize)]
pub struct ProfileRequest {
    /// Discord user id.
    pub user_id: String,
    /// Guild id.
    pub tenant_id: String,
}

/// What the engine remembers about one user.
#[derive(Debug, Default, Deserialize)]
pub struct ProfileList {
    /// Things remembered.
    #[serde(default)]
    pub nodes: Vec<ProfileNode>,
    /// How those things relate.
    #[serde(default)]
    pub relations: Vec<ProfileRelation>,
}

/// One remembered thing.
#[derive(Debug, Clone, Deserialize)]
pub struct ProfileNode {
    /// Category, such as course or club.
    pub kind: String,
    /// Name of the thing.
    pub label: String,
    /// Belief from 0 to 1.
    pub confidence: f64,
}

/// One remembered relation between two things.
#[derive(Debug, Clone, Deserialize)]
pub struct ProfileRelation {
    /// Label of the first thing.
    pub subject: String,
    /// How they relate.
    pub relation: String,
    /// Label of the second thing.
    pub object: String,
    /// Belief from 0 to 1.
    pub confidence: f64,
}

/// Body of POST /profile/forget. No label removes everything.
#[derive(Debug, Serialize)]
pub struct ForgetRequest {
    /// Discord user id.
    pub user_id: String,
    /// Guild id.
    pub tenant_id: String,
    /// The one thing to forget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Reply to POST /profile/forget.
#[derive(Debug, Deserialize)]
pub struct ForgetResponse {
    /// Items removed.
    pub removed: u64,
}

/// One PostHog product event.
#[derive(Debug, Clone, Serialize)]
pub struct AnalyticsEvent {
    /// Event name, such as discord_ask.
    pub event: &'static str,
    /// The Discord user id.
    pub distinct_id: String,
    /// Event properties.
    pub properties: serde_json::Map<String, serde_json::Value>,
    /// When it happened, RFC 3339 in UTC.
    pub timestamp: String,
}

impl AnalyticsEvent {
    /// An event named event for distinct_id, stamped now, with no properties.
    pub fn new(event: &'static str, distinct_id: &impl ToString) -> Self {
        Self {
            event,
            distinct_id: distinct_id.to_string(),
            properties: serde_json::Map::new(),
            timestamp: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        }
    }

    /// Adds one property.
    #[must_use]
    pub fn with(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.properties.insert(key.to_owned(), value.into());
        self
    }
}

/// Body of POST /batch/ on PostHog.
#[derive(Debug, Serialize)]
pub struct AnalyticsBatch<'a> {
    /// Project token.
    pub api_key: &'a str,
    /// The events.
    pub batch: &'a [AnalyticsEvent],
}
