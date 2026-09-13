//! ProfileGraph trait. Every query is scoped to the tenant and user of the request.

use async_trait::async_trait;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::memory::profile::{
    ProfileError, ProfileFact, ProfileNode, ProfileRelation,
};

/// The entity graph for one user in one tenant. Scope comes from the request context alone.
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

    /// The relations already recorded for this subject and relation.
    async fn matching(
        &self,
        ctx: &RequestContext,
        subject: &str,
        relation: &str,
    ) -> Result<Vec<ProfileRelation>, ProfileError>;

    /// Removes one relation. Returns whether it was there.
    async fn drop_relation(
        &self,
        ctx: &RequestContext,
        relation: &ProfileRelation,
    ) -> Result<bool, ProfileError>;

    /// Removes one node and every relation through it. Returns how many nodes went.
    async fn forget(&self, ctx: &RequestContext, label: &str) -> Result<u64, ProfileError>;

    /// Removes everything this user carries: every node and every memories row. Returns how
    /// many rows went.
    async fn forget_all(&self, ctx: &RequestContext) -> Result<u64, ProfileError>;
}
