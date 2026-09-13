//! PostgreSQL adapter for the profile graph: profile_nodes and profile_edges over one pool.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

use crate::core::traits::knowledge::retrieval::Embedder;
use crate::core::traits::memory::profile::ProfileGraph;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::memory::profile::{
    ProfileEntity, ProfileError, ProfileFact, ProfileNode, ProfileRelation,
};
use crate::stores::postgres::{row_limit, vector_literal};

/// Maps a sqlx error to a ProfileError.
#[allow(clippy::needless_pass_by_value)]
fn db(e: sqlx::Error) -> ProfileError {
    ProfileError::Store(e.to_string())
}

/// The entity graph over profile_nodes and profile_edges.
pub struct PgProfileGraph {
    pool: PgPool,
    embedder: Arc<dyn Embedder>,
}

impl PgProfileGraph {
    /// Builds the graph over a pool and the embedder the index was built with.
    pub fn new(pool: PgPool, embedder: Arc<dyn Embedder>) -> Self {
        Self { pool, embedder }
    }

    /// Embeds every distinct label in facts, keyed by kind and label.
    async fn vectors(
        &self,
        facts: &[ProfileFact],
    ) -> Result<HashMap<(String, String), Vec<f32>>, ProfileError> {
        let mut keys: Vec<(String, String)> = Vec::new();
        for fact in facts {
            for entity in [&fact.subject, &fact.object] {
                let key = (entity.kind.clone(), entity.label.clone());
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
        }
        let texts: Vec<String> = keys
            .iter()
            .map(|(kind, label)| format!("{kind}: {label}"))
            .collect();
        let vectors = self
            .embedder
            .embed(&texts)
            .await
            .map_err(|e| ProfileError::Embedding(e.to_string()))?;
        if vectors.len() != keys.len() {
            return Err(ProfileError::Embedding(format!(
                "asked for {} vectors and got {}",
                keys.len(),
                vectors.len()
            )));
        }
        for vector in &vectors {
            if vector.len() != self.embedder.dim() {
                return Err(ProfileError::Embedding(format!(
                    "embedding has {} dimensions; the graph holds {}",
                    vector.len(),
                    self.embedder.dim()
                )));
            }
        }
        Ok(keys.into_iter().zip(vectors).collect())
    }
}

/// Reads one relation row selected with subject_kind, subject_label, relation, object_kind,
/// object_label, and confidence.
fn row_to_relation(row: &sqlx::postgres::PgRow) -> Result<ProfileRelation, ProfileError> {
    Ok(ProfileRelation {
        subject: ProfileEntity {
            kind: row.try_get("subject_kind").map_err(db)?,
            label: row.try_get("subject_label").map_err(db)?,
        },
        relation: row.try_get("relation").map_err(db)?,
        object: ProfileEntity {
            kind: row.try_get("object_kind").map_err(db)?,
            label: row.try_get("object_label").map_err(db)?,
        },
        confidence: row.try_get("confidence").map_err(db)?,
    })
}

/// Inserts or refreshes one node and returns its id.
async fn node_id(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    tenant_id: &str,
    user_id: Uuid,
    entity: &ProfileEntity,
    embedding: &str,
    confidence: f32,
) -> Result<Uuid, ProfileError> {
    let row = sqlx::query(
        "insert into profile_nodes (tenant_id, user_id, kind, label, embedding, confidence)
         values ($1, $2, $3, $4, $5::vector, $6)
         on conflict (tenant_id, user_id, kind, label) do update
           set embedding = excluded.embedding,
               confidence = greatest(profile_nodes.confidence, excluded.confidence),
               updated_at = now()
         returning id",
    )
    .bind(tenant_id)
    .bind(user_id)
    .bind(&entity.kind)
    .bind(&entity.label)
    .bind(embedding)
    .bind(confidence)
    .fetch_one(&mut **tx)
    .await
    .map_err(db)?;
    row.try_get("id").map_err(db)
}

#[async_trait]
impl ProfileGraph for PgProfileGraph {
    async fn upsert(
        &self,
        ctx: &RequestContext,
        facts: &[ProfileFact],
    ) -> Result<(), ProfileError> {
        if facts.is_empty() {
            return Ok(());
        }
        let vectors = self.vectors(facts).await?;
        let mut tx = self.pool.begin().await.map_err(db)?;
        let user_id: Uuid = sqlx::query_scalar(
            "insert into users (tenant_id, discord_id, roles) values ($1, $2, $3)
             on conflict (tenant_id, discord_id) do update set roles = excluded.roles
             returning id",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(&ctx.roles)
        .fetch_one(&mut *tx)
        .await
        .map_err(db)?;

        for fact in facts {
            let mut ends = Vec::with_capacity(2);
            for entity in [&fact.subject, &fact.object] {
                let key = (entity.kind.clone(), entity.label.clone());
                let Some(vector) = vectors.get(&key) else {
                    return Err(ProfileError::Embedding(format!(
                        "no vector for {}: {}",
                        entity.kind, entity.label
                    )));
                };
                ends.push(
                    node_id(
                        &mut tx,
                        &ctx.tenant_id,
                        user_id,
                        entity,
                        &vector_literal(vector),
                        fact.confidence,
                    )
                    .await?,
                );
            }
            let [from_node, to_node] = ends[..] else {
                return Err(ProfileError::Store("a fact has two ends".into()));
            };
            sqlx::query(
                "insert into profile_edges (tenant_id, from_node, to_node, relation, confidence)
                 values ($1, $2, $3, $4, $5)
                 on conflict (from_node, to_node, relation) do update
                   set confidence = greatest(profile_edges.confidence, excluded.confidence)",
            )
            .bind(&ctx.tenant_id)
            .bind(from_node)
            .bind(to_node)
            .bind(&fact.relation)
            .bind(fact.confidence)
            .execute(&mut *tx)
            .await
            .map_err(db)?;
        }
        tx.commit().await.map_err(db)
    }

    async fn recall(
        &self,
        ctx: &RequestContext,
        limit: usize,
    ) -> Result<Vec<ProfileNode>, ProfileError> {
        let rows = sqlx::query(
            "select n.id, n.kind, n.label, n.confidence, n.created_at, n.updated_at
             from profile_nodes n join users u on u.id = n.user_id
             where n.tenant_id = $1 and u.discord_id = $2
             order by n.confidence desc, n.updated_at desc limit $3",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(row_limit(limit))
        .fetch_all(&self.pool)
        .await
        .map_err(db)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            out.push(ProfileNode {
                id: row.try_get("id").map_err(db)?,
                kind: row.try_get("kind").map_err(db)?,
                label: row.try_get("label").map_err(db)?,
                confidence: row.try_get("confidence").map_err(db)?,
                created_at: row.try_get("created_at").map_err(db)?,
                updated_at: row.try_get("updated_at").map_err(db)?,
            });
        }
        Ok(out)
    }

    async fn relations(
        &self,
        ctx: &RequestContext,
        limit: usize,
    ) -> Result<Vec<ProfileRelation>, ProfileError> {
        let rows = sqlx::query(
            "select s.kind as subject_kind, s.label as subject_label, e.relation,
                    o.kind as object_kind, o.label as object_label, e.confidence
             from profile_edges e
             join profile_nodes s on s.id = e.from_node
             join profile_nodes o on o.id = e.to_node
             join users u on u.id = s.user_id
             where e.tenant_id = $1 and u.discord_id = $2
             order by e.confidence desc, e.created_at desc limit $3",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(row_limit(limit))
        .fetch_all(&self.pool)
        .await
        .map_err(db)?;
        rows.iter().map(row_to_relation).collect()
    }

    async fn matching(
        &self,
        ctx: &RequestContext,
        subject: &str,
        relation: &str,
    ) -> Result<Vec<ProfileRelation>, ProfileError> {
        let rows = sqlx::query(
            "select s.kind as subject_kind, s.label as subject_label, e.relation,
                    o.kind as object_kind, o.label as object_label, e.confidence
             from profile_edges e
             join profile_nodes s on s.id = e.from_node
             join profile_nodes o on o.id = e.to_node
             join users u on u.id = s.user_id
             where e.tenant_id = $1 and u.discord_id = $2
               and lower(s.label) = lower($3) and lower(e.relation) = lower($4)
             order by e.created_at",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(subject)
        .bind(relation)
        .fetch_all(&self.pool)
        .await
        .map_err(db)?;
        rows.iter().map(row_to_relation).collect()
    }

    async fn drop_relation(
        &self,
        ctx: &RequestContext,
        relation: &ProfileRelation,
    ) -> Result<bool, ProfileError> {
        // Deletes the edge and leaves both nodes.
        let done = sqlx::query(
            "delete from profile_edges e
             using profile_nodes s, profile_nodes o, users u
             where e.from_node = s.id and e.to_node = o.id and s.user_id = u.id
               and e.tenant_id = $1 and u.discord_id = $2
               and lower(s.label) = lower($3) and lower(e.relation) = lower($4)
               and lower(o.label) = lower($5)",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(&relation.subject.label)
        .bind(&relation.relation)
        .bind(&relation.object.label)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(done.rows_affected() > 0)
    }

    async fn forget(&self, ctx: &RequestContext, label: &str) -> Result<u64, ProfileError> {
        // Edges cascade with the node.
        let done = sqlx::query(
            "delete from profile_nodes n using users u
             where n.user_id = u.id and n.tenant_id = $1 and u.discord_id = $2 and n.label = $3",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(label)
        .execute(&self.pool)
        .await
        .map_err(db)?;
        Ok(done.rows_affected())
    }

    async fn forget_all(&self, ctx: &RequestContext) -> Result<u64, ProfileError> {
        let mut tx = self.pool.begin().await.map_err(db)?;
        let nodes = sqlx::query(
            "delete from profile_nodes n using users u
             where n.user_id = u.id and n.tenant_id = $1 and u.discord_id = $2",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        let memories = sqlx::query(
            "delete from memories m using users u
             where m.user_id = u.id and m.tenant_id = $1 and u.discord_id = $2",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .execute(&mut *tx)
        .await
        .map_err(db)?;
        tx.commit().await.map_err(db)?;
        Ok(nodes.rows_affected() + memories.rows_affected())
    }
}
