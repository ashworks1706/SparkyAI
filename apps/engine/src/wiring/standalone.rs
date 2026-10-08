//! Standalone composition: PostgreSQL stores, the retrieval index, the Redis query cache, OAuth.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::Stores;
use crate::core::config::Config;
use crate::core::traits::knowledge::admission::Admission;
use crate::core::traits::knowledge::cache::QueryCache;
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::traits::knowledge::retrieval::Embedder;
use crate::core::traits::memory::profile::ProfileGraph;
use crate::core::traits::oauth::OAuthStore;
use crate::core::traits::trace::TraceSink;
use crate::core::types::tools::oauth::OAuthError;
use crate::runtime::harness::knowledge::admit::AdmittedQueries;
use crate::runtime::harness::knowledge::cache::{CacheRules, CachedQueries};
use crate::runtime::tools::account::oauth::WebOAuthClient;
use crate::runtime::tools::knowledge::search;
use crate::stores::standalone::health::PgProbe;
use crate::stores::standalone::knowledge::cache::{
    self as redis_cache, RedisAdmission, RedisQueryCache,
};
use crate::stores::standalone::memory::profile::PgProfileGraph;
use crate::stores::standalone::oauth::PgOAuth;
use crate::stores::standalone::postgres::{
    self, PgConfirmations, PgConversations, PgMemory, PgRetriever, PgSourceQueries, RetrievalTuning,
};

/// Every store over this engine's PostgreSQL and Redis. Both must be reachable at boot.
pub(super) async fn stores(
    cfg: &Config,
    embedder: &Arc<dyn Embedder>,
    trace: &Arc<dyn TraceSink>,
) -> anyhow::Result<Stores> {
    let Some(url) = &cfg.postgres.url else {
        anyhow::bail!("set SPARKY_POSTGRES__URL, or turn on platform.enabled");
    };
    let pool = postgres::connect(
        url,
        cfg.postgres.max_connections,
        Duration::from_secs(cfg.postgres.acquire_timeout_secs),
    )
    .await
    .map_err(|e| anyhow::anyhow!("postgres: {e}"))?;
    let profile_graph = cfg.profile.enabled.then(|| {
        Arc::new(PgProfileGraph::new(pool.clone(), Arc::clone(embedder))) as Arc<dyn ProfileGraph>
    });
    Ok(Stores {
        retriever: Arc::new(PgRetriever::new(
            pool.clone(),
            Arc::clone(embedder),
            RetrievalTuning::from(&cfg.retrieval),
        )),
        queries: source_queries(cfg, &pool, Arc::clone(trace)).await?,
        conversations: Arc::new(PgConversations::new(pool.clone())),
        memory: Arc::new(PgMemory::new(pool.clone())),
        confirmations: Arc::new(PgConfirmations::new(pool.clone())),
        profile_graph,
        oauth_store: Arc::new(PgOAuth::new(pool.clone())) as Arc<dyn OAuthStore>,
        oauth_providers: oauth_providers(cfg)?,
        probes: vec![Arc::new(PgProbe::new(pool))],
    })
}

/// Builds the web OAuth client of one provider.
type OAuthBuild<'a> = Box<dyn FnOnce() -> Result<WebOAuthClient, OAuthError> + 'a>;

/// A web OAuth client per enabled provider, for the consent routes and token refresh.
fn oauth_providers(cfg: &Config) -> anyhow::Result<HashMap<String, Arc<WebOAuthClient>>> {
    let mut providers: HashMap<String, Arc<WebOAuthClient>> = HashMap::new();
    let clients: [(&str, bool, OAuthBuild<'_>); 3] = [
        (
            "canvas",
            cfg.oauth.canvas.enabled,
            Box::new(|| WebOAuthClient::canvas(&cfg.oauth.canvas)),
        ),
        (
            "microsoft",
            cfg.oauth.microsoft.enabled,
            Box::new(|| WebOAuthClient::microsoft(&cfg.oauth.microsoft)),
        ),
        (
            "google",
            cfg.oauth.google.enabled,
            Box::new(|| WebOAuthClient::google(&cfg.oauth.google)),
        ),
    ];
    for (provider, enabled, build) in clients {
        if !enabled {
            continue;
        }
        let client = build().map_err(|e| anyhow::anyhow!("oauth.{provider}: {e}"))?;
        providers.insert(provider.to_owned(), Arc::new(client));
    }
    Ok(providers)
}

/// The registry and queue the scraper serves, behind the shared cache when one is configured.
async fn source_queries(
    cfg: &Config,
    pool: &sqlx::PgPool,
    trace: Arc<dyn TraceSink>,
) -> anyhow::Result<Arc<dyn SourceQueries>> {
    let mut queries: Arc<dyn SourceQueries> = Arc::new(PgSourceQueries::new(
        pool.clone(),
        Duration::from_millis(cfg.query.poll_ms),
        Duration::from_millis(cfg.query.poll_max_ms),
        Duration::from_secs(cfg.query.claim_secs),
    ));
    let caching = cfg.tools.search && cfg.query.cache.enabled;
    let capping = cfg.tools.search && cfg.query.max_in_flight > 0;
    if !(caching || capping) {
        tracing::info!("the live query cache and cap are off; every query is fetched");
        return Ok(queries);
    }
    // Config::validate rejects either of them without a redis section.
    let Some(redis) = &cfg.redis else {
        return Ok(queries);
    };
    let conn = redis_cache::connect(&redis.url, Duration::from_secs(redis.connect_timeout_secs))
        .await
        .map_err(|e| anyhow::anyhow!("redis: {e}"))?;
    let call_budget = Duration::from_millis(cfg.query.cache.timeout_ms);

    // The cap wraps first and the cache sits above it, so a cache hit takes no slot.
    if capping {
        let admission: Arc<dyn Admission> = Arc::new(RedisAdmission::new(
            conn.clone(),
            call_budget,
            "sparky:query:v1:in-flight",
            cfg.query.max_in_flight,
            Duration::from_secs(cfg.query.cache.lease_secs),
        ));
        tracing::info!(limit = cfg.query.max_in_flight, "live query cap registered");
        queries = Arc::new(AdmittedQueries::new(queries, admission, Arc::clone(&trace)));
    }
    if caching {
        let cache: Arc<dyn QueryCache> = Arc::new(RedisQueryCache::new(conn, call_budget));
        let rules = cache_rules(cfg);
        let reused = rules.ttl.values().filter(|ttl| !ttl.is_zero()).count();
        tracing::info!(
            sources = rules.ttl.len(),
            reused,
            "live query cache registered"
        );
        queries = Arc::new(CachedQueries::new(queries, cache, trace, rules));
    }
    Ok(queries)
}

/// How long each source's answers are reused: the per-source setting, else the one for its kind.
fn cache_rules(cfg: &Config) -> CacheRules {
    let settings = &cfg.query.cache;
    let mut ttl = std::collections::HashMap::new();
    for source in search::catalog() {
        let default = if source.freshness().indexed() {
            settings.default_ttl_secs
        } else {
            settings.live_ttl_secs
        };
        let secs = settings
            .ttl_secs
            .get(source.key())
            .copied()
            .unwrap_or(default);
        ttl.insert(source.key().to_owned(), Duration::from_secs(secs));
    }
    CacheRules {
        ttl,
        handoff: Duration::from_secs(settings.handoff_secs),
        lease: Duration::from_secs(settings.lease_secs),
        poll: Duration::from_millis(settings.poll_ms),
        ignore_words: settings
            .ignore_words
            .iter()
            .map(|word| word.to_lowercase())
            .collect(),
        keep_words: settings.keep_words.iter().cloned().collect(),
    }
}
