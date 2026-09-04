//! `MemoryStore` trait. Recall is always scoped to tenant and user.

use async_trait::async_trait;
use uuid::Uuid;

use crate::core::types::context::RequestContext;
use crate::core::types::memory::{Memory, MemoryCandidate, MemoryQuery};
use crate::core::types::store::StoreError;

/// Cross-conversation memory for one user in one tenant.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Recalls unexpired memories, newest and most confident first.
    async fn recall(
        &self,
        ctx: &RequestContext,
        q: &MemoryQuery,
    ) -> Result<Vec<Memory>, StoreError>;
    /// Writes a candidate if policy admits it; returns the new id.
    async fn write(
        &self,
        ctx: &RequestContext,
        m: &MemoryCandidate,
    ) -> Result<Option<Uuid>, StoreError>;
    /// Deletes one memory the user owns.
    async fn forget(&self, ctx: &RequestContext, id: Uuid) -> Result<(), StoreError>;
}
