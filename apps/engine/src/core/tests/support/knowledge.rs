//! Knowledge doubles: a fixed source query registry, an index of canned hits, a query cache.

use async_trait::async_trait;

use crate::core::traits::knowledge::cache::QueryCache;
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::traits::knowledge::retrieval::Retriever;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::cache::{CacheError, Entry};
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::knowledge::query::{
    QueryError, QueryOutcome, QueryRequest, QuerySourceInfo,
};
use crate::core::types::knowledge::retrieval::{RetrievalError, RetrievalQuery};

/// A Retriever double: one canned hit, and the queries it was asked, readable after it moves.
pub struct Stored {
    hits: Vec<Evidence>,
    asked: std::sync::Arc<std::sync::Mutex<Vec<RetrievalQuery>>>,
}

impl Stored {
    /// An index holding one passage of text under title.
    pub fn holding(title: &str, content: &str) -> Self {
        Self {
            hits: vec![Evidence {
                source_id: uuid::Uuid::nil(),
                chunk_id: uuid::Uuid::nil(),
                title: title.to_owned(),
                content: content.to_owned(),
                url: Some(format!("https://example.test/{title}")),
                fetched_at: chrono::Utc::now() - chrono::Duration::hours(3),
                score: 1.0,
            }],
            asked: std::sync::Arc::default(),
        }
    }

    /// An index holding nothing.
    pub fn empty() -> Self {
        Self {
            hits: Vec::new(),
            asked: std::sync::Arc::default(),
        }
    }

    /// The queries this double is asked, readable after it moves into a tool.
    pub fn asked(&self) -> std::sync::Arc<std::sync::Mutex<Vec<RetrievalQuery>>> {
        std::sync::Arc::clone(&self.asked)
    }
}

#[async_trait]
impl Retriever for Stored {
    async fn retrieve(
        &self,
        _ctx: &RequestContext,
        query: &RetrievalQuery,
    ) -> Result<Vec<Evidence>, RetrievalError> {
        if let Ok(mut asked) = self.asked.lock() {
            asked.push(query.clone());
        }
        Ok(self.hits.clone())
    }
}

/// A SourceQueries double: a fixed registry and a canned outcome per source.
pub struct FakeQueries {
    sources: Vec<QuerySourceInfo>,
    answers: std::collections::HashMap<String, Result<QueryOutcome, String>>,
    sent: std::sync::Arc<std::sync::Mutex<Vec<QueryRequest>>>,
    takes: std::time::Duration,
}

impl FakeQueries {
    /// Offers sources and rejects anything not given an answer.
    pub fn new(sources: Vec<QuerySourceInfo>) -> Self {
        Self {
            sources,
            answers: std::collections::HashMap::new(),
            sent: std::sync::Arc::default(),
            takes: std::time::Duration::ZERO,
        }
    }

    /// Makes every fetch take this long, so two requests for one query overlap.
    pub fn taking(mut self, takes: std::time::Duration) -> Self {
        self.takes = takes;
        self
    }

    /// The requests this double is sent, readable after it moves into a tool.
    pub fn sent(&self) -> std::sync::Arc<std::sync::Mutex<Vec<QueryRequest>>> {
        std::sync::Arc::clone(&self.sent)
    }

    /// Answers source with this page text.
    pub fn answering(mut self, source: &str, text: &str) -> Self {
        self.answers.insert(
            source.to_owned(),
            Ok(QueryOutcome {
                source: source.to_owned(),
                url: format!("https://example.test/{source}"),
                text: text.to_owned(),
                fetched_at: None,
            }),
        );
        self
    }

    /// Rejects source with this reason.
    pub fn rejecting(mut self, source: &str, reason: &str) -> Self {
        self.answers
            .insert(source.to_owned(), Err(reason.to_owned()));
        self
    }
}

#[async_trait]
impl SourceQueries for FakeQueries {
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError> {
        Ok(self.sources.clone())
    }

    async fn run(
        &self,
        _ctx: &RequestContext,
        request: &QueryRequest,
    ) -> Result<QueryOutcome, QueryError> {
        if let Ok(mut sent) = self.sent.lock() {
            sent.push(request.clone());
        }
        if !self.takes.is_zero() {
            tokio::time::sleep(self.takes).await;
        }
        match self.answers.get(&request.source) {
            Some(Ok(outcome)) => Ok(outcome.clone()),
            Some(Err(reason)) => Err(QueryError::Rejected(reason.clone())),
            None => Err(QueryError::Rejected(format!(
                "unknown source {:?}",
                request.source
            ))),
        }
    }
}

/// A QueryCache double: an in-memory map with lifetimes the test controls.
#[derive(Default)]
pub struct FakeCache {
    entries: std::sync::Mutex<std::collections::HashMap<String, (Entry, std::time::Instant)>>,
    /// Calls that fail instead of answering, for the path where the cache is not there.
    broken: bool,
}

impl FakeCache {
    /// A working cache holding nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// A cache that fails every call, as an unreachable Redis does.
    pub fn broken() -> Self {
        Self {
            broken: true,
            ..Self::default()
        }
    }

    /// How many keys hold something that has not expired.
    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or_default()
    }

    /// Whether the cache holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn fail<T>(what: &str) -> Result<T, CacheError> {
        Err(CacheError::Backend(format!("{what}: no cache here")))
    }
}

#[async_trait]
impl QueryCache for FakeCache {
    async fn get(&self, key: &str) -> Result<Option<Entry>, CacheError> {
        if self.broken {
            return Self::fail("get");
        }
        let Ok(mut entries) = self.entries.lock() else {
            return Ok(None);
        };
        let expired = entries
            .get(key)
            .is_some_and(|(_, until)| *until <= std::time::Instant::now());
        if expired {
            entries.remove(key);
        }
        Ok(entries.get(key).map(|(entry, _)| entry.clone()))
    }

    async fn claim(&self, key: &str, lease: std::time::Duration) -> Result<bool, CacheError> {
        if self.broken {
            return Self::fail("claim");
        }
        let Ok(mut entries) = self.entries.lock() else {
            return Ok(false);
        };
        let held = entries
            .get(key)
            .is_some_and(|(_, until)| *until > std::time::Instant::now());
        if held {
            return Ok(false);
        }
        entries.insert(
            key.to_owned(),
            (Entry::Pending, std::time::Instant::now() + lease),
        );
        Ok(true)
    }

    async fn put(
        &self,
        key: &str,
        entry: &Entry,
        ttl: std::time::Duration,
    ) -> Result<(), CacheError> {
        if self.broken {
            return Self::fail("put");
        }
        if let Ok(mut entries) = self.entries.lock() {
            entries.insert(
                key.to_owned(),
                (entry.clone(), std::time::Instant::now() + ttl),
            );
        }
        Ok(())
    }

    async fn release(&self, key: &str) -> Result<(), CacheError> {
        if self.broken {
            return Self::fail("release");
        }
        if let Ok(mut entries) = self.entries.lock() {
            entries.remove(key);
        }
        Ok(())
    }
}
