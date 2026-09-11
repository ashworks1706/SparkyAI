//! Confirmation store doubles: one held action.

use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;

use crate::core::traits::safety::confirmation::ConfirmationStore;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::safety::policy::PendingAction;
use crate::core::types::store::StoreError;

/// Holds one action, the way the database does: single use, and only for who was asked.
#[derive(Default)]
pub struct Held {
    held: Mutex<Option<(uuid::Uuid, String, PendingAction)>>,
}

impl Held {
    pub fn holds(&self) -> bool {
        self.held.lock().is_ok_and(|h| h.is_some())
    }
}

#[async_trait]
impl ConfirmationStore for Held {
    async fn hold(
        &self,
        ctx: &RequestContext,
        token: uuid::Uuid,
        pending: &PendingAction,
        _payload_hash: &str,
        _ttl: Duration,
    ) -> Result<(), StoreError> {
        if let Ok(mut slot) = self.held.lock() {
            *slot = Some((token, ctx.user_id.clone(), pending.clone()));
        }
        Ok(())
    }

    async fn claim(
        &self,
        ctx: &RequestContext,
        token: uuid::Uuid,
        _approved: bool,
    ) -> Result<Option<PendingAction>, StoreError> {
        let Ok(mut slot) = self.held.lock() else {
            return Ok(None);
        };
        match slot.as_ref() {
            Some((held, asked, _)) if *held == token && *asked == ctx.user_id => {
                Ok(slot.take().map(|(_, _, pending)| pending))
            }
            _ => Ok(None),
        }
    }
}
