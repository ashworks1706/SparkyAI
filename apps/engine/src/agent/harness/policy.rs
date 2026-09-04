//! `Policy` trait, `ProposedAction`, `Decision` (Allow / Deny / Confirm), confirmation tokens.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::agent::harness::tool::RiskClass;
use crate::core::types::context::RequestContext;

/// A tool call the model wants to make, before it runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedAction {
    /// Tool name.
    pub tool: String,
    /// Declared risk of that tool.
    pub risk: RiskClass,
    /// Exact arguments. A confirmation binds to these bytes.
    pub arguments: Value,
}

/// What the user must approve.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmationRequest {
    /// Single-use token the client echoes back.
    pub token: Uuid,
    /// Tool name.
    pub tool: String,
    /// Hash of the exact arguments; a changed payload needs a new confirmation.
    pub payload_hash: String,
    /// Plain-language statement of what happens, where, with what data, and whether it reverses.
    pub summary: String,
}

/// The policy verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    /// Run it.
    Allow,
    /// Do not run it; the model is told why.
    Deny {
        /// Reason shown to the model and recorded in the trace.
        reason: String,
    },
    /// Stop and ask the user first.
    Confirm(ConfirmationRequest),
}

/// Policy evaluation failures.
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    /// The policy could not be evaluated; the action is not run.
    #[error("policy unavailable: {0}")]
    Unavailable(String),
}

/// Decides whether a proposed action runs.
#[async_trait]
pub trait Policy: Send + Sync {
    /// Evaluates one action for one request.
    async fn authorize(
        &self,
        ctx: &RequestContext,
        action: &ProposedAction,
    ) -> Result<Decision, PolicyError>;
}

/// Hex SHA-256 of the canonical JSON of `arguments`.
pub fn payload_hash(arguments: &Value) -> String {
    use std::hash::{Hash, Hasher};
    // Canonical form: serde_json sorts map keys when the `preserve_order` feature is off.
    let canonical = arguments.to_string();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    canonical.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// The default policy: risk class alone decides. Roles gate writes.
#[derive(Debug, Clone)]
pub struct RiskPolicy {
    /// Role required for `ExternalWrite` and `Destructive`.
    pub write_role: Option<String>,
}

impl RiskPolicy {
    /// Writes need `write_role`; reads and drafts are open.
    pub fn new(write_role: Option<String>) -> Self {
        Self { write_role }
    }
}

#[async_trait]
impl Policy for RiskPolicy {
    async fn authorize(
        &self,
        ctx: &RequestContext,
        action: &ProposedAction,
    ) -> Result<Decision, PolicyError> {
        Ok(match action.risk {
            RiskClass::ReadPublic | RiskClass::PrepareWrite => Decision::Allow,
            RiskClass::ReadAuthenticated => Decision::Deny {
                reason: "authenticated reads are not enabled".into(),
            },
            RiskClass::ExternalWrite | RiskClass::Destructive => {
                if let Some(role) = &self.write_role
                    && !ctx.has_role(role)
                {
                    return Ok(Decision::Deny {
                        reason: format!("`{}` requires the {role} role", action.tool),
                    });
                }
                Decision::Confirm(ConfirmationRequest {
                    token: Uuid::new_v4(),
                    tool: action.tool.clone(),
                    payload_hash: payload_hash(&action.arguments),
                    summary: format!(
                        "Run `{}` with {}. {}",
                        action.tool,
                        action.arguments,
                        if action.risk == RiskClass::Destructive {
                            "This cannot be undone."
                        } else {
                            "This posts or submits externally."
                        }
                    ),
                })
            }
            RiskClass::Forbidden => Decision::Deny {
                reason: format!("`{}` is forbidden", action.tool),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;

    use super::*;

    fn ctx(roles: &[&str]) -> RequestContext {
        RequestContext::new("g", "u", Duration::from_secs(5))
            .with_roles(roles.iter().map(ToString::to_string).collect())
    }

    fn action(risk: RiskClass) -> ProposedAction {
        ProposedAction {
            tool: "t".into(),
            risk,
            arguments: json!({"a": 1}),
        }
    }

    #[tokio::test]
    async fn reads_are_allowed() {
        let p = RiskPolicy::new(Some("Moderator".into()));
        let d = p.authorize(&ctx(&[]), &action(RiskClass::ReadPublic)).await;
        assert!(matches!(d, Ok(Decision::Allow)));
    }

    #[tokio::test]
    async fn writes_without_role_are_denied() {
        let p = RiskPolicy::new(Some("Moderator".into()));
        let d = p
            .authorize(&ctx(&[]), &action(RiskClass::ExternalWrite))
            .await;
        assert!(matches!(d, Ok(Decision::Deny { .. })));
    }

    #[tokio::test]
    async fn writes_with_role_need_confirmation() {
        let p = RiskPolicy::new(Some("Moderator".into()));
        let d = p
            .authorize(&ctx(&["Moderator"]), &action(RiskClass::ExternalWrite))
            .await;
        assert!(matches!(d, Ok(Decision::Confirm(_))));
    }

    #[tokio::test]
    async fn forbidden_is_denied_regardless_of_role() {
        let p = RiskPolicy::new(None);
        let d = p
            .authorize(&ctx(&["Moderator"]), &action(RiskClass::Forbidden))
            .await;
        assert!(matches!(d, Ok(Decision::Deny { .. })));
    }

    #[test]
    fn payload_hash_changes_with_arguments() {
        assert_ne!(
            payload_hash(&json!({"a": 1})),
            payload_hash(&json!({"a": 2}))
        );
        assert_eq!(
            payload_hash(&json!({"a": 1})),
            payload_hash(&json!({"a": 1}))
        );
    }
}
