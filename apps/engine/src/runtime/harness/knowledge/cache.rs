//! Tool output caching for live queries: reuse a recent answer, and never fetch one twice at once.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use crate::core::traits::knowledge::cache::QueryCache;
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::cache::{CacheOutcome, Entry};
use crate::core::types::knowledge::query::{
    QueryError, QueryOutcome, QueryRequest, QuerySourceInfo,
};
use crate::core::types::trace::TraceEvent;

/// Namespace of every query cache key, so a key format change never reads an older entry.
const KEY_NAMESPACE: Uuid = Uuid::from_u128(0x5f8a_7c21_9e4b_4d13_8a6f_2c05_b391_7de4);

/// Prefix of every key written to the shared cache.
const KEY_PREFIX: &str = "sparky:query:v1:";

/// How long each source's answers may be reused, and the timings of the lease.
#[derive(Debug, Clone)]
pub struct CacheRules {
    /// Longest an answer of each source is reused, before the handoff floor applies.
    pub ttl: HashMap<String, Duration>,
    /// Floor under every source's lifetime: long enough for the requests that waited to read it.
    pub handoff: Duration,
    /// Longest one request may hold the lease before another may take it.
    pub lease: Duration,
    /// How often a request waiting on the lease looks for the answer.
    pub poll: Duration,
    /// Lowercase words left out of a text parameter when it is keyed.
    pub ignore_words: HashSet<String>,
    /// Sources whose text parameters keep every word in the key.
    pub keep_words: HashSet<String>,
}

impl CacheRules {
    /// How long an answer of source is reused. handoff is the floor, so this is never zero:
    /// an answer has to outlive the fetch for the requests that waited on it to read it.
    fn lifetime(&self, source: &str) -> Duration {
        self.handoff
            .max(self.ttl.get(source).copied().unwrap_or_default())
    }
}

/// The key one request and params map to, the same for every caller in a tenant.
fn key(tenant_id: &str, request: &QueryRequest, ignore: &HashSet<String>) -> String {
    let mut params: Vec<(&String, String)> = request
        .params
        .iter()
        .map(|(name, value)| match value.as_str() {
            Some(text) => (name, keyed_text(text, ignore)),
            None => (name, value.to_string()),
        })
        .collect();
    params.sort();
    let mut canonical = format!("{tenant_id}\u{1f}{}", request.source);
    for (name, value) in params {
        canonical.push('\u{1f}');
        canonical.push_str(name);
        canonical.push('=');
        canonical.push_str(&value);
    }
    format!(
        "{KEY_PREFIX}{}",
        Uuid::new_v5(&KEY_NAMESPACE, canonical.as_bytes())
    )
}

/// A text parameter as it is keyed: lowercase words, punctuation dropped, ignored words left out.
fn keyed_text(text: &str, ignore: &HashSet<String>) -> String {
    let lower = text.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty() && !ignore.contains(*word))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Live queries over a shared cache: one fetch per query, and answers reused within their lifetime.
pub struct CachedQueries {
    inner: Arc<dyn SourceQueries>,
    cache: Arc<dyn QueryCache>,
    trace: Arc<dyn TraceSink>,
    rules: CacheRules,
}

impl CachedQueries {
    /// Wraps queries in the cache. A source not named in rules.ttl is reused for handoff only.
    pub fn new(
        inner: Arc<dyn SourceQueries>,
        cache: Arc<dyn QueryCache>,
        trace: Arc<dyn TraceSink>,
        rules: CacheRules,
    ) -> Self {
        Self {
            inner,
            cache,
            trace,
            rules,
        }
    }

    /// What the cache holds, or None when it holds nothing or could not be reached.
    async fn stored(&self, key: &str) -> Option<Entry> {
        match self.cache.get(key).await {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(%error, "query cache read failed; fetching instead");
                None
            }
        }
    }

    /// An entry as the answer it stands for, with the age of a reused one shown to the model.
    fn answer(entry: Entry) -> Option<Result<QueryOutcome, QueryError>> {
        match entry {
            Entry::Answer {
                mut outcome,
                fetched_at,
            } => {
                outcome.fetched_at = Some(fetched_at);
                Some(Ok(outcome))
            }
            Entry::Refusal { reason } => Some(Err(QueryError::Rejected(reason))),
            Entry::Pending => None,
        }
    }

    /// Fetches, then stores what came back. A failure that is not the source's leaves no lease.
    async fn fetch(
        &self,
        ctx: &RequestContext,
        request: &QueryRequest,
        key: &str,
    ) -> Result<QueryOutcome, QueryError> {
        let result = self.inner.run(ctx, request).await;
        let entry = match &result {
            Ok(outcome) => Entry::Answer {
                outcome: outcome.clone(),
                fetched_at: Utc::now(),
            },
            // A refusal is about the arguments, so it holds as briefly as an answer a waiter reads.
            Err(QueryError::Rejected(reason)) => Entry::Refusal {
                reason: reason.clone(),
            },
            // Everything else says nothing about the query, so the next request starts over.
            Err(_) => {
                if let Err(error) = self.cache.release(key).await {
                    tracing::warn!(%error, "query cache lease was not released");
                }
                return result;
            }
        };
        let lifetime = match &entry {
            Entry::Answer { .. } => self.rules.lifetime(&request.source),
            _ => self.rules.handoff,
        };

        if let Err(error) = self.cache.put(key, &entry, lifetime).await {
            tracing::warn!(%error, "query result was not cached");
        }
        result
    }

    /// Waits for whoever holds the lease. None means the caller should fetch it itself.
    async fn wait(
        &self,
        ctx: &RequestContext,
        key: &str,
    ) -> Option<Result<QueryOutcome, QueryError>> {
        loop {
            if ctx.cancel.is_cancelled() {
                return Some(Err(QueryError::Cancelled));
            }
            let remaining = ctx.remaining();
            if remaining.is_zero() {
                return Some(Err(QueryError::Timeout(remaining)));
            }
            tokio::time::sleep(self.rules.poll.min(remaining)).await;
            match self.stored(key).await {
                Some(entry) => {
                    if let Some(answer) = Self::answer(entry) {
                        return Some(answer);
                    }
                }
                // The lease expired or was released, so this request fetches it instead.
                None => return None,
            }
        }
    }
}

#[async_trait]
impl SourceQueries for CachedQueries {
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError> {
        self.inner.sources().await
    }

    async fn run(
        &self,
        ctx: &RequestContext,
        request: &QueryRequest,
    ) -> Result<QueryOutcome, QueryError> {
        let none = HashSet::new();
        let ignore = if self.rules.keep_words.contains(&request.source) {
            &none
        } else {
            &self.rules.ignore_words
        };
        let key = key(&ctx.tenant_id, request, ignore);
        let mut outcome = CacheOutcome::Miss;
        let report = |outcome| {
            self.trace.emit(
                ctx,
                TraceEvent::QueryCache {
                    source: request.source.clone(),
                    outcome,
                },
            );
        };

        // A stored answer within its lifetime answers straight away.
        if let Some(entry) = self.stored(&key).await
            && let Some(answer) = Self::answer(entry)
        {
            report(CacheOutcome::Hit);
            return answer;
        }

        // Otherwise exactly one request fetches and the rest read what it wrote.
        loop {
            match self.cache.claim(&key, self.rules.lease).await {
                Ok(true) => break,
                // None means the lease went away, so the next turn of the loop takes it.
                Ok(false) => {
                    if let Some(answer) = self.wait(ctx, &key).await {
                        report(CacheOutcome::Coalesced);
                        return answer;
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "query cache claim failed; fetching instead");
                    outcome = CacheOutcome::Unavailable;
                    break;
                }
            }
        }
        report(outcome);
        self.fetch(ctx, request, &key).await
    }
}
