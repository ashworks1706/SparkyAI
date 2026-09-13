//! MemoryStore trait. Recall is always scoped to tenant and user.

pub mod detector;
pub mod profile;

use async_trait::async_trait;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::memory::{Memory, MemoryQuery};
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
}
