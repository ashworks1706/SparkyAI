//! Buttons the bot puts on its own messages, and the `custom_id` that identifies them.
//!
//! A component is described here as a `ButtonSpec` and turned into serenity's builders at the
//! edge, so what a message offers can be decided and tested without a Discord connection.
//! Adding a button is a new `Action`, a row that includes it, and an arm in the bot's dispatch.

use serenity::all::{ButtonStyle, CreateActionRow, CreateButton};
use uuid::Uuid;

use crate::core::types::ChatResponse;

/// Marks a `custom_id` as this bot's, so a component from anywhere else is ignored.
const PREFIX: &str = "sparky";

/// What a button does when someone presses it.
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

/// Everything a pressed button has to tell the bot. Discord caps `custom_id` at 100 bytes, so
/// the two ids are written without dashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CustomId {
    /// What to do.
    pub action: Action,
    /// The confirmation being answered.
    pub token: Uuid,
    /// The conversation it belongs to.
    pub conversation: Uuid,
}

impl CustomId {
    /// Builds an id for one button.
    pub fn new(action: Action, token: Uuid, conversation: Uuid) -> Self {
        Self {
            action,
            token,
            conversation,
        }
    }

    /// Reads an id back, or `None` when the bot did not mint it.
    pub fn parse(raw: &str) -> Option<Self> {
        let mut parts = raw.split(':');
        if parts.next()? != PREFIX {
            return None;
        }
        let action = Action::parse(parts.next()?)?;
        let token = Uuid::parse_str(parts.next()?).ok()?;
        let conversation = Uuid::parse_str(parts.next()?).ok()?;
        if parts.next().is_some() {
            return None;
        }
        Some(Self {
            action,
            token,
            conversation,
        })
    }
}

impl std::fmt::Display for CustomId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{PREFIX}:{}:{}:{}",
            self.action.as_str(),
            self.token.simple(),
            self.conversation.simple()
        )
    }
}

/// One button, described without serenity so it can be decided and tested on its own.
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
            label: "Approve",
            danger: true,
        },
        ButtonSpec {
            id: CustomId::new(Action::Deny, confirmation.token, resp.conversation_id),
            label: "Cancel",
            danger: false,
        },
    ];
    vec![row]
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
