//! The default `RiskPolicy` and payload hashing for confirmations.

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use crate::core::traits::policy::Policy;
use crate::core::types::context::RequestContext;
use crate::core::types::policy::{ConfirmationRequest, Decision, ProposedAction};
use crate::core::types::tool::RiskClass;

/// Stable hash of the canonical JSON of `arguments`. A changed payload needs a new confirmation.
pub fn payload_hash(arguments: &Value) -> String {
    use std::hash::{Hash, Hasher};
    // serde_json sorts map keys when `preserve_order` is off, so this is canonical.
    let canonical = arguments.to_string();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    canonical.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// The default risk policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct RiskPolicy;

impl RiskPolicy {
    /// Creates the policy.
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Policy for RiskPolicy {
    async fn authorize(&self, ctx: &RequestContext, action: &ProposedAction) -> Decision {
        match action.risk {
            RiskClass::ReadPublic | RiskClass::PrepareWrite => Decision::Allow,
            RiskClass::ReadAuthenticated => Decision::Deny {
                reason: "authenticated reads are not enabled".into(),
            },
            RiskClass::ExternalWrite | RiskClass::Destructive => {
                if !ctx.has_role("MANAGE_GUILD") {
                    return Decision::Deny {
                        reason: format!(
                            "`{}` requires Discord's Manage Server permission",
                            action.tool
                        ),
                    };
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
        }
    }
}
