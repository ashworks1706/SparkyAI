//! Memory doubles: a memory store and a profile graph that count their recalls.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use crate::core::traits::memory::MemoryStore;
use crate::core::traits::memory::profile::ProfileGraph;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::memory::profile::{
    ProfileEntity, ProfileError, ProfileFact, ProfileNode, ProfileRelation,
};
use crate::core::types::memory::{Memory, MemoryKind, MemoryQuery};
use crate::core::types::store::StoreError;

/// A memory store holding one memory, counting how often it is asked.
#[derive(Default)]
pub struct Recalling {
    calls: AtomicUsize,
}

impl Recalling {
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl MemoryStore for Recalling {
    async fn recall(
        &self,
        _ctx: &RequestContext,
        _q: &MemoryQuery,
    ) -> Result<Vec<Memory>, StoreError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(vec![Memory {
            id: uuid::Uuid::nil(),
            kind: MemoryKind::Semantic,
            content: "studies CSE 310".into(),
            confidence: 1.0,
            created_at: chrono::Utc::now(),
            expires_at: None,
        }])
    }
}

/// A profile graph holding one node and one relation, counting recalls.
#[derive(Default)]
pub struct Known {
    recalls: AtomicUsize,
}

impl Known {
    pub fn recalls(&self) -> usize {
        self.recalls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl ProfileGraph for Known {
    async fn upsert(
        &self,
        _ctx: &RequestContext,
        _facts: &[ProfileFact],
    ) -> Result<(), ProfileError> {
        Ok(())
    }

    async fn recall(
        &self,
        _ctx: &RequestContext,
        _limit: usize,
    ) -> Result<Vec<ProfileNode>, ProfileError> {
        self.recalls.fetch_add(1, Ordering::Relaxed);
        Ok(vec![ProfileNode {
            id: uuid::Uuid::nil(),
            kind: "course".into(),
            label: "CSE 310".into(),
            confidence: 0.9,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }])
    }

    async fn relations(
        &self,
        _ctx: &RequestContext,
        _limit: usize,
    ) -> Result<Vec<ProfileRelation>, ProfileError> {
        Ok(vec![ProfileRelation {
            subject: ProfileEntity {
                kind: "person".into(),
                label: "the user".into(),
            },
            relation: "studies".into(),
            object: ProfileEntity {
                kind: "course".into(),
                label: "CSE 310".into(),
            },
            confidence: 0.8,
        }])
    }

    async fn matching(
        &self,
        _ctx: &RequestContext,
        _subject: &str,
        _relation: &str,
    ) -> Result<Vec<ProfileRelation>, ProfileError> {
        Ok(Vec::new())
    }

    async fn drop_relation(
        &self,
        _ctx: &RequestContext,
        _relation: &ProfileRelation,
    ) -> Result<bool, ProfileError> {
        Ok(false)
    }

    async fn forget(&self, _ctx: &RequestContext, _label: &str) -> Result<u64, ProfileError> {
        Ok(0)
    }

    async fn forget_all(&self, _ctx: &RequestContext) -> Result<u64, ProfileError> {
        Ok(0)
    }
}
