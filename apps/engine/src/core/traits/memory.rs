//! `MemoryStore` trait. Recall is always scoped to tenant and user.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::memory::{Memory, MemoryQuery};
use crate::core::types::store::StoreError;

/// Cross-conversation memory for one user in one tenant. Writes arrive with Phase 4.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Recalls unexpired memories, newest and most confident first.
    async fn recall(
        &self,
        ctx: &RequestContext,
        q: &MemoryQuery,
    ) -> Result<Vec<Memory>, StoreError>;
}
