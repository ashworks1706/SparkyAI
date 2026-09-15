//! Admission: how many live queries may reach the database at once, across every engine replica.

use async_trait::async_trait;

use crate::core::types::knowledge::cache::CacheError;

/// A cap on live queries in flight. Holders that never leave are dropped when their lease expires.
#[async_trait]
pub trait Admission: Send + Sync {
    /// Takes a slot for holder. False means every slot is taken.
    async fn enter(&self, holder: &str) -> Result<bool, CacheError>;

    /// Gives back the slot of holder.
    async fn leave(&self, holder: &str) -> Result<(), CacheError>;
}
