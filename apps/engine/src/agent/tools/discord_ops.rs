//! `ExternalWrite` Discord operations: announcements, polls, tickets, escalation. Confirmation-gated.
//! Phase 3 wires these to the bot; today they declare their schema and risk so policy and
//! confirmation can be exercised end to end.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::core::traits::tool::Tool;
use crate::core::types::context::RequestContext;
use crate::core::types::tool::{RiskClass, ToolDefinition, ToolError, ToolOutput};

/// Posts an announcement to a channel.
pub struct PostAnnouncement;

#[async_trait]
impl Tool for PostAnnouncement {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "post_announcement".into(),
            description: "Post an announcement to a Discord channel. Moderators only; the user \
                          confirms before it is sent."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "channel": { "type": "string" },
                    "text": { "type": "string" }
                },
                "required": ["channel", "text"]
            }),
            risk: RiskClass::ExternalWrite,
        }
    }

    async fn call(&self, _ctx: &RequestContext, _args: Value) -> Result<ToolOutput, ToolError> {
        Err(ToolError::Failed(
            "Discord write operations arrive in Phase 3".into(),
        ))
    }
}
