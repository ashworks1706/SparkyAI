//! The profile graph on the platform agents routes. The platform embeds nodes itself.

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Value, json};

use super::client::{Call, PlatformClient, PlatformError, field, timestamp};
use crate::core::traits::memory::profile::ProfileGraph;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::memory::profile::{
    ProfileError, ProfileFact, ProfileNode, ProfileRelation,
};

/// Maps a platform error to a ProfileError.
#[allow(clippy::needless_pass_by_value)]
fn store(e: PlatformError) -> ProfileError {
    ProfileError::Store(e.to_string())
}

/// The profile graph held by the platform agents module.
pub struct PlatformProfileGraph {
    client: PlatformClient,
}

impl PlatformProfileGraph {
    /// Wraps a client.
    pub fn new(client: PlatformClient) -> Self {
        Self { client }
    }

    /// Sends one call to a member route and returns the body.
    async fn send(
        &self,
        ctx: &RequestContext,
        method: Method,
        rest: &str,
        query: &[(&str, String)],
        body: Option<&Value>,
    ) -> Result<Value, ProfileError> {
        let path = PlatformClient::member(&ctx.user_id, rest);
        let mut call = Call::new(method, &path).query(query);
        if let Some(body) = body {
            call = call.body(body);
        }
        self.client.call(&call).await.map_err(store)
    }
}

/// One node from its platform row.
fn node(row: &Value) -> Result<ProfileNode, PlatformError> {
    let confidence: f64 = field(row, "confidence")?;
    // Confidence is a fraction from 0 to 1.
    #[allow(clippy::cast_possible_truncation)]
    Ok(ProfileNode {
        id: field(row, "id")?,
        kind: field(row, "kind")?,
        label: field(row, "label")?,
        confidence: confidence as f32,
        created_at: timestamp(row, "created_at")?,
        updated_at: timestamp(row, "updated_at")?,
    })
}

#[async_trait]
impl ProfileGraph for PlatformProfileGraph {
    async fn upsert(
        &self,
        ctx: &RequestContext,
        facts: &[ProfileFact],
    ) -> Result<(), ProfileError> {
        if facts.is_empty() {
            return Ok(());
        }
        let body = json!({ "facts": facts });
        self.send(ctx, Method::POST, "/profile/facts", &[], Some(&body))
            .await?;
        Ok(())
    }

    async fn recall(
        &self,
        ctx: &RequestContext,
        limit: usize,
    ) -> Result<Vec<ProfileNode>, ProfileError> {
        let query = [("limit", limit.max(1).to_string())];
        let value = self
            .send(ctx, Method::GET, "/profile", &query, None)
            .await?;
        let rows: Vec<Value> = field(&value, "nodes").map_err(store)?;
        rows.iter().map(|r| node(r).map_err(store)).collect()
    }

    async fn relations(
        &self,
        ctx: &RequestContext,
        limit: usize,
    ) -> Result<Vec<ProfileRelation>, ProfileError> {
        let query = [("limit", limit.max(1).to_string())];
        let value = self
            .send(ctx, Method::GET, "/profile", &query, None)
            .await?;
        field(&value, "relations").map_err(store)
    }

    async fn matching(
        &self,
        ctx: &RequestContext,
        subject: &str,
        relation: &str,
    ) -> Result<Vec<ProfileRelation>, ProfileError> {
        let query = [
            ("subject", subject.to_owned()),
            ("relation", relation.to_owned()),
        ];
        let value = self
            .send(ctx, Method::GET, "/profile/matching", &query, None)
            .await?;
        field(&value, "relations").map_err(store)
    }

    async fn drop_relation(
        &self,
        ctx: &RequestContext,
        relation: &ProfileRelation,
    ) -> Result<bool, ProfileError> {
        let body = json!({
            "subject": relation.subject.label,
            "relation": relation.relation,
            "object": relation.object.label,
        });
        let value = self
            .send(ctx, Method::DELETE, "/profile/relations", &[], Some(&body))
            .await?;
        field(&value, "deleted").map_err(store)
    }

    async fn forget(&self, ctx: &RequestContext, label: &str) -> Result<u64, ProfileError> {
        let query = [("label", label.to_owned())];
        let value = self
            .send(ctx, Method::DELETE, "/profile/nodes", &query, None)
            .await?;
        field(&value, "deleted").map_err(store)
    }

    async fn forget_all(&self, ctx: &RequestContext) -> Result<u64, ProfileError> {
        let value = self.send(ctx, Method::DELETE, "/data", &[], None).await?;
        field(&value, "deleted").map_err(store)
    }
}
