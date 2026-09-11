//! Buttons the bot puts on its own messages, and the custom_id that identifies them.
//!
//! A component is described here as a ButtonSpec and turned into serenity builders at the edge.
//! Adding a button is a CustomId variant, a row that includes it, and an arm in the bot dispatch.

use serenity::all::{ButtonStyle, CreateActionRow, CreateButton};
use uuid::Uuid;

use crate::core::types::ChatResponse;

/// Marks a custom_id as belonging to this bot. A component from anywhere else is ignored.
const PREFIX: &str = "sparky";
/// Tag of the button that erases everything remembered.
const FORGET_ALL: &str = "forget_all";
/// Tag of the button that closes the forget prompt.
const KEEP_ALL: &str = "keep_all";

/// What a confirmation button does when someone presses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Run the held action and let the agent carry on.
    Approve,
    /// Drop the held action.
    Deny,
}

impl Action {
    fn as_str(self) -> &'static str {
        match self {
            Self::Approve => "approve",
            Self::Deny => "deny",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "approve" => Some(Self::Approve),
            "deny" => Some(Self::Deny),
            _ => None,
        }
    }
}

/// Everything a pressed button has to tell the bot. Discord caps custom_id at 100 bytes. Uuids
/// are written without dashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomId {
    /// Answers a held action.
    Confirm {
        /// What to do.
        action: Action,
        /// The confirmation being answered.
        token: Uuid,
        /// The conversation it belongs to.
        conversation: Uuid,
    },
    /// Erases everything remembered about the user who asked.
    ForgetAll {
        /// Discord id of the user who asked.
        user: u64,
    },
    /// Closes the forget prompt without erasing anything.
    KeepAll {
        /// Discord id of the user who asked.
        user: u64,
    },
}

impl CustomId {
    /// Builds an id for one confirmation button.
    pub fn new(action: Action, token: Uuid, conversation: Uuid) -> Self {
        Self::Confirm {
            action,
            token,
            conversation,
        }
    }

    /// Whether presser may press this button. Forget buttons belong to the user who asked;
    /// the engine checks the caller of a confirmation itself.
    pub fn may_press(&self, presser: u64) -> bool {
        match self {
            Self::Confirm { .. } => true,
            Self::ForgetAll { user } | Self::KeepAll { user } => *user == presser,
        }
    }

    /// Reads an id back, or None when the bot did not mint it.
    pub fn parse(raw: &str) -> Option<Self> {
        let mut parts = raw.split(':');
        if parts.next()? != PREFIX {
            return None;
        }
        let id = match parts.next()? {
            FORGET_ALL => Self::ForgetAll {
                user: parts.next()?.parse().ok()?,
            },
            KEEP_ALL => Self::KeepAll {
                user: parts.next()?.parse().ok()?,
            },
            other => Self::Confirm {
                action: Action::parse(other)?,
                token: Uuid::parse_str(parts.next()?).ok()?,
                conversation: Uuid::parse_str(parts.next()?).ok()?,
            },
        };
        if parts.next().is_some() {
            return None;
        }
        Some(id)
    }
}

impl std::fmt::Display for CustomId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Confirm {
                action,
                token,
                conversation,
            } => write!(
                f,
                "{PREFIX}:{}:{}:{}",
                action.as_str(),
                token.simple(),
                conversation.simple()
            ),
            Self::ForgetAll { user } => write!(f, "{PREFIX}:{FORGET_ALL}:{user}"),
            Self::KeepAll { user } => write!(f, "{PREFIX}:{KEEP_ALL}:{user}"),
        }
    }
}

/// One button, described without serenity types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ButtonSpec {
    /// What pressing it means.
    pub id: CustomId,
    /// Text on the button.
    pub label: &'static str,
    /// Whether it reads as the safe or the consequential answer.
    pub danger: bool,
}

/// The buttons a reply should carry. Empty when it asks nothing of the reader.
pub fn rows_for(resp: &ChatResponse) -> Vec<Vec<ButtonSpec>> {
    let Some(confirmation) = &resp.confirmation else {
        return Vec::new();
    };
    let row = vec![
        ButtonSpec {
            id: CustomId::new(Action::Approve, confirmation.token, resp.conversation_id),
            label: "Yes, do it",
            danger: true,
        },
        ButtonSpec {
            id: CustomId::new(Action::Deny, confirmation.token, resp.conversation_id),
            label: "No",
            danger: false,
        },
    ];
    vec![row]
}

/// The buttons under the forget everything prompt, bound to the user who asked.
pub fn forget_rows(user: u64) -> Vec<Vec<ButtonSpec>> {
    vec![vec![
        ButtonSpec {
            id: CustomId::ForgetAll { user },
            label: "Forget everything",
            danger: true,
        },
        ButtonSpec {
            id: CustomId::KeepAll { user },
            label: "Cancel",
            danger: false,
        },
    ]]
}

/// Turns described rows into what serenity sends.
pub fn to_action_rows(rows: &[Vec<ButtonSpec>]) -> Vec<CreateActionRow> {
    rows.iter()
        .map(|row| {
            CreateActionRow::Buttons(
                row.iter()
                    .map(|b| {
                        CreateButton::new(b.id.to_string())
                            .label(b.label)
                            .style(if b.danger {
                                ButtonStyle::Danger
                            } else {
                                ButtonStyle::Secondary
                            })
                    })
                    .collect(),
            )
        })
        .collect()
}
