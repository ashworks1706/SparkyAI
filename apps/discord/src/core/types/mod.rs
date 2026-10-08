//! Wire types mirrored from the engine HTTP contract, plus the client error, by domain.

pub mod analytics;
pub mod chat;
pub mod conversation;
pub mod error;
pub mod oauth;
pub mod profile;
pub mod stream;

pub use analytics::AnalyticsEvent;
pub use chat::{ChatRequest, ChatResponse, ConfirmRequest};
pub use conversation::{Attachment, FileAttachment, ResetRequest, ResetResponse, Visibility};
pub use error::EngineError;
pub use oauth::{AuthorizeRequest, AuthorizeResponse, DisconnectResponse};
pub use profile::{ForgetRequest, ForgetResponse, ProfileList, ProfileRequest};
pub use stream::{ErrorFrame, Progress, Update};
