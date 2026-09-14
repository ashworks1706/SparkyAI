//! QueryCache: the shared store that holds live query answers and the leases over them.

use std::time::Duration;

use async_trait::async_trait;

use crate::core::types::knowledge::cache::{CacheError, Entry};

/// A shared store of live query answers, keyed by the query that produced them.
#[async_trait]
pub trait QueryCache: Send + Sync {
    /// What is stored for key, or None when nothing is.
    async fn get(&self, key: &str) -> Result<Option<Entry>, CacheError>;

    /// Stores Pending for key only if nothing is stored. True when this caller took the lease.
    async fn claim(&self, key: &str, lease: Duration) -> Result<bool, CacheError>;

    /// Stores entry for key for ttl, replacing whatever is there.
    async fn put(&self, key: &str, entry: &Entry, ttl: Duration) -> Result<(), CacheError>;

    /// Removes key, so a fetch that did not finish leaves no lease behind.
    async fn release(&self, key: &str) -> Result<(), CacheError>;
}
