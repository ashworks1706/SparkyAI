//! Memories table.

pub mod profile;

use async_trait::async_trait;
use sqlx::Row;
use sqlx::postgres::PgPool;

use crate::core::traits::memory::MemoryStore;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::memory::{Memory, MemoryKind, MemoryQuery};
use crate::core::types::store::StoreError;
use crate::stores::postgres::{db, row_limit};

/// Memories table. Every query is scoped by tenant and user.
pub struct PgMemory {
    pool: PgPool,
}

impl PgMemory {
    /// Wraps a pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl MemoryStore for PgMemory {
    async fn recall(
        &self,
        ctx: &RequestContext,
        q: &MemoryQuery,
    ) -> Result<Vec<Memory>, StoreError> {
        let kinds: Vec<&str> = q.kinds.iter().map(|k| k.as_str()).collect();
        let rows = sqlx::query(
            "select m.id, m.kind, m.content, m.confidence, m.created_at, m.expires_at
             from memories m join users u on u.id = m.user_id
             where m.tenant_id = $1 and u.discord_id = $2
               and (cardinality($3::text[]) = 0 or m.kind = any($3))
               and (m.expires_at is null or m.expires_at > now())
             order by m.confidence desc, m.created_at desc limit $4",
        )
        .bind(&ctx.tenant_id)
        .bind(&ctx.user_id)
        .bind(&kinds)
        .bind(row_limit(q.limit))
        .fetch_all(&self.pool)
        .await
        .map_err(db)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let kind: String = row.try_get("kind").map_err(db)?;
            let Some(kind) = MemoryKind::parse(&kind) else {
                return Err(StoreError::Database(format!(
                    "memories.kind {kind:?} is not a kind this build knows; schema and code \
                     disagree"
                )));
            };
            out.push(Memory {
                id: row.try_get("id").map_err(db)?,
                kind,
                content: row.try_get("content").map_err(db)?,
                confidence: row.try_get("confidence").map_err(db)?,
                created_at: row.try_get("created_at").map_err(db)?,
                expires_at: row.try_get("expires_at").map_err(db)?,
            });
        }
        Ok(out)
    }
}
