//! Readiness of the PostgreSQL pool.

use async_trait::async_trait;
use sqlx::postgres::PgPool;

use crate::core::traits::health::Probe;

/// Answers readiness with select 1 on the pool.
pub struct PgProbe {
    pool: PgPool,
}

impl PgProbe {
    /// Wraps a pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl Probe for PgProbe {
    fn name(&self) -> &'static str {
        "postgres"
    }

    async fn ready(&self) -> bool {
        match sqlx::query("select 1").execute(&self.pool).await {
            Ok(_) => true,
            Err(error) => {
                tracing::warn!(%error, "readiness: postgres did not answer");
                false
            }
        }
    }
}
