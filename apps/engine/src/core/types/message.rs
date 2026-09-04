//! Message, tool-call, and tool-result types exchanged with the model.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions from the operator.
    System,
    /// The end user.
    User,
    /// The model.
    Assistant,
    /// A tool result, answering one `ToolCall`.
    Tool,
}

/// A tool invocation the model asked for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned id, echoed back in the matching `Role::Tool` message.
    pub id: String,
    /// Tool name, matching a `ToolDefinition`.
    pub name: String,
    /// Arguments as parsed JSON.
    pub arguments: Value,
}

/// One turn in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who produced it.
    pub role: Role,
    /// Text content. Empty when an assistant turn is only tool calls.
    pub content: String,
    /// Tool calls requested by an assistant turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For `Role::Tool`: the `ToolCall::id` this answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// A system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::plain(Role::System, content)
    }

    /// A user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::plain(Role::User, content)
    }

    /// An assistant message with text only.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::plain(Role::Assistant, content)
    }

    /// An assistant turn carrying tool calls.
    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls,
            tool_call_id: None,
        }
    }

    /// A tool result answering `call_id`.
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
        }
    }

    fn plain(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// Rough token estimate used for budgeting before a request is sent.
    pub fn estimated_tokens(&self) -> usize {
        let call_chars: usize = self
            .tool_calls
            .iter()
            .map(|c| c.name.len() + c.arguments.to_string().len())
            .sum();
        (self.content.len() + call_chars) / 4 + 4
    }
}
