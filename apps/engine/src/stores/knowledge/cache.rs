//! The Redis query cache: answers, refusals, and the leases that keep one fetch per query.

use std::time::Duration;

use async_trait::async_trait;
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use secrecy::{ExposeSecret, SecretString};

use crate::core::traits::knowledge::cache::QueryCache;
use crate::core::types::knowledge::cache::{CacheError, Entry};

/// Live query answers shared by every engine replica pointed at the same Redis.
pub struct RedisQueryCache {
    conn: ConnectionManager,
    budget: Duration,
}

/// Connects to Redis, or fails with what could not be reached.
pub async fn connect(
    url: &SecretString,
    budget: Duration,
) -> Result<ConnectionManager, CacheError> {
    let client = redis::Client::open(url.expose_secret())
        .map_err(|e| CacheError::Backend(format!("redis url: {e}")))?;
    let conn = tokio::time::timeout(budget, client.get_connection_manager())
        .await
        .map_err(|_| CacheError::Backend(format!("connect timed out after {budget:?}")))?
        .map_err(|e| CacheError::Backend(e.to_string()))?;
    Ok(conn)
}

impl RedisQueryCache {
    /// Builds the cache over a connection; budget bounds one call.
    pub fn new(conn: ConnectionManager, budget: Duration) -> Self {
        Self { conn, budget }
    }

    /// Runs one command inside the call budget.
    async fn within<T, F>(&self, what: &str, call: F) -> Result<T, CacheError>
    where
        F: std::future::Future<Output = redis::RedisResult<T>>,
    {
        tokio::time::timeout(self.budget, call)
            .await
            .map_err(|_| CacheError::Backend(format!("{what} timed out after {:?}", self.budget)))?
            .map_err(|e| CacheError::Backend(format!("{what}: {e}")))
    }
}

/// Seconds a TTL is worth, never zero: Redis takes no expiry below one second.
fn seconds(ttl: Duration) -> u64 {
    ttl.as_secs().max(1)
}

#[async_trait]
impl QueryCache for RedisQueryCache {
    async fn get(&self, key: &str) -> Result<Option<Entry>, CacheError> {
        let mut conn = self.conn.clone();
        let stored: Option<String> = self.within("get", conn.get(key)).await?;
        let Some(stored) = stored else {
            return Ok(None);
        };
        // An entry written by an older format is a miss, not a failure.
        match serde_json::from_str(&stored) {
            Ok(entry) => Ok(Some(entry)),
            Err(error) => {
                tracing::warn!(%error, key, "unreadable cache entry; treating it as a miss");
                Ok(None)
            }
        }
    }

    async fn claim(&self, key: &str, lease: Duration) -> Result<bool, CacheError> {
        let pending = serde_json::to_string(&Entry::Pending)
            .map_err(|e| CacheError::Backend(format!("encode pending: {e}")))?;
        let mut conn = self.conn.clone();
        let options = redis::SetOptions::default()
            .conditional_set(redis::ExistenceCheck::NX)
            .with_expiration(redis::SetExpiry::EX(seconds(lease)));
        let taken: Option<String> = self
            .within("claim", conn.set_options(key, pending, options))
            .await?;
        Ok(taken.is_some())
    }

    async fn put(&self, key: &str, entry: &Entry, ttl: Duration) -> Result<(), CacheError> {
        let encoded = serde_json::to_string(entry)
            .map_err(|e| CacheError::Backend(format!("encode entry: {e}")))?;
        let mut conn = self.conn.clone();
        self.within("put", conn.set_ex::<_, _, ()>(key, encoded, seconds(ttl)))
            .await
    }

    async fn release(&self, key: &str) -> Result<(), CacheError> {
        let mut conn = self.conn.clone();
        self.within("release", conn.del::<_, ()>(key)).await
    }
}
