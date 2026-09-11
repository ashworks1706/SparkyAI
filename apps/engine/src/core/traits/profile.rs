//! ProfileGraph trait. Every query is scoped to the tenant and user of the request.

use async_trait::async_trait;

use crate::core::types::context::RequestContext;
use crate::core::types::profile::{ProfileError, ProfileFact, ProfileNode, ProfileRelation};

/// The entity graph for one user in one tenant.
///
/// Scope comes from the request context alone, so no caller can name another user's graph.
#[async_trait]
pub trait ProfileGraph: Send + Sync {
    /// Writes facts as nodes and the edges between them, updating what is already there.
    async fn upsert(&self, ctx: &RequestContext, facts: &[ProfileFact])
    -> Result<(), ProfileError>;

    /// Returns this user's nodes, most confident and most recently confirmed first.
    async fn recall(
        &self,
        ctx: &RequestContext,
        limit: usize,
    ) -> Result<Vec<ProfileNode>, ProfileError>;

    /// The relations this user carries, most confident first.
    async fn relations(
        &self,
        ctx: &RequestContext,
        limit: usize,
    ) -> Result<Vec<ProfileRelation>, ProfileError>;

    /// Removes one node and every relation through it. Returns how many nodes went.
    async fn forget(&self, ctx: &RequestContext, label: &str) -> Result<u64, ProfileError>;

    /// Removes everything this user carries. Returns how many nodes went.
    async fn forget_all(&self, ctx: &RequestContext) -> Result<u64, ProfileError>;
}
