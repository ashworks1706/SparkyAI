//! The tool set: search, sandbox, MCP servers, and the enabled integrations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::core::config::Config;
use crate::core::traits::knowledge::admission::Admission;
use crate::core::traits::knowledge::cache::QueryCache;
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::traits::knowledge::retrieval::Retriever;
use crate::core::traits::oauth::OAuthStore;
use crate::core::traits::tools::Tool;
use crate::core::traits::trace::TraceSink;
use crate::core::types::tools::sandbox::Limits as SandboxLimits;
use crate::runtime::harness::knowledge::admit::AdmittedQueries;
use crate::runtime::harness::knowledge::cache::{CacheRules, CachedQueries};
use crate::runtime::harness::tools::ToolSet;
use crate::runtime::tools::account::oauth::WebOAuthClient;
use crate::runtime::tools::account::{canvas, gcal, outlook};
use crate::runtime::tools::knowledge::search;
use crate::runtime::tools::knowledge::search::live::{LiveSearch, Wording as LiveWording};
use crate::runtime::tools::knowledge::search::stored::{StoredSearch, Wording as StoredWording};
use crate::runtime::tools::mcp::{self, McpLimits};
use crate::runtime::tools::sandbox::{
    ContainerSandbox, SandboxTool, Wording as SandboxWording, reap_sessions,
};
use crate::runtime::tools::{papers, transit, wiki};
use crate::stores::knowledge::cache::{self as redis_cache, RedisAdmission, RedisQueryCache};
use crate::stores::postgres::PgSourceQueries;

/// Builds the tools of one integration, or says why it could not.
type Build<'a> = Box<dyn FnOnce() -> Result<Vec<Arc<dyn Tool>>, String> + 'a>;

/// One integration: its name, whether it is enabled, and how its tools are built.
type Integration<'a> = (&'static str, bool, Build<'a>);

/// Every tool the model may call, with tools.disabled removed at registration.
pub(super) async fn build_tools(
    cfg: &Config,
    queries: Arc<dyn SourceQueries>,
    retriever: Arc<dyn Retriever>,
    sandbox: Option<Arc<ContainerSandbox>>,
    oauth_store: Arc<dyn OAuthStore>,
    oauth_providers: HashMap<String, Arc<WebOAuthClient>>,
) -> anyhow::Result<(ToolSet, Vec<String>)> {
    let disabled = |name: &str| cfg.tools.disabled.iter().any(|d| d == name);
    let mut tools = ToolSet::new();
    let mut mcp_names = Vec::new();
    if cfg.tools.search {
        for tool in search_tools(cfg, queries, retriever).await? {
            if !disabled(&tool.definition().name) {
                tools = tools.with(tool);
            }
        }
    }
    if let Some(sandbox) = sandbox {
        let tool: Arc<dyn Tool> = Arc::new(SandboxTool::new(
            sandbox,
            cfg.sandbox.risk,
            SandboxWording::from(&cfg.sandbox),
        ));
        if !disabled(&tool.definition().name) {
            tracing::info!(
                runtime = %cfg.sandbox.runtime,
                image = %cfg.sandbox.image,
                "sandbox registered"
            );
            tools = tools.with(tool);
        }
    }
    for server in cfg.mcp.resolved_servers() {
        let limits = McpLimits {
            required_props_only: server
                .required_props_only
                .unwrap_or(cfg.mcp.required_props_only),
            tool_timeout_secs: server.tool_timeout_secs,
            ..McpLimits::from(&cfg.mcp)
        };
        let remote = mcp::connect(&server.url, &server.tools, &server.risks, &limits)
            .await
            .map_err(|e| anyhow::anyhow!("mcp server {} at {}: {e}", server.name, server.url))?;
        let mut registered = 0;
        for tool in remote {
            if disabled(&tool.definition().name) {
                continue;
            }
            mcp_names.push(tool.definition().name);
            tools = tools.with(tool);
            registered += 1;
        }
        tracing::info!(
            server = %server.name,
            url = %server.url,
            count = registered,
            "mcp tools registered"
        );
    }
    for (integration, enabled, build) in integrations(cfg, &oauth_store, &oauth_providers) {
        if !enabled {
            continue;
        }
        let built = build().map_err(|e| anyhow::anyhow!("{integration}: {e}"))?;
        tools = register(tools, built, &disabled, integration);
    }
    tracing::info!(tools = ?tools, "tool set");
    Ok((tools, mcp_names))
}

/// Every integration that adds tools, in registration order.
fn integrations<'a>(
    cfg: &'a Config,
    oauth_store: &'a Arc<dyn OAuthStore>,
    oauth_providers: &'a HashMap<String, Arc<WebOAuthClient>>,
) -> [Integration<'a>; 6] {
    [
        (
            "canvas",
            cfg.canvas.enabled,
            Box::new(move || {
                canvas::tools(
                    &cfg.canvas,
                    oauth_store.clone(),
                    oauth_providers.get("canvas").cloned(),
                )
                .map_err(|e| e.to_string())
            }),
        ),
        (
            "outlook",
            cfg.outlook.enabled,
            Box::new(move || {
                outlook::tools(
                    &cfg.outlook,
                    oauth_store.clone(),
                    oauth_providers.get("microsoft").cloned(),
                )
                .map_err(|e| e.to_string())
            }),
        ),
        (
            "gcal",
            cfg.gcal.enabled,
            Box::new(move || {
                gcal::tools(
                    &cfg.gcal,
                    oauth_store.clone(),
                    oauth_providers.get("google").cloned(),
                )
                .map_err(|e| e.to_string())
            }),
        ),
        (
            "papers",
            cfg.papers.enabled,
            Box::new(move || papers::tools(&cfg.papers)),
        ),
        (
            "wikipedia",
            cfg.wikipedia.enabled,
            Box::new(move || wiki::tools(&cfg.wikipedia)),
        ),
        (
            "transit",
            cfg.transit.enabled,
            Box::new(move || transit::tools(&cfg.transit)),
        ),
    ]
}

/// Adds each built tool that is not disabled to the set, logging how many an integration added.
fn register(
    mut tools: ToolSet,
    built: Vec<Arc<dyn Tool>>,
    disabled: &dyn Fn(&str) -> bool,
    integration: &str,
) -> ToolSet {
    let mut count = 0;
    for tool in built {
        if disabled(&tool.definition().name) {
            continue;
        }
        tools = tools.with(tool);
        count += 1;
    }
    tracing::info!(integration, count, "tools registered");
    tools
}

/// The stored and live search tools, after checking the catalog against the published registry.
async fn search_tools(
    cfg: &Config,
    queries: Arc<dyn SourceQueries>,
    retriever: Arc<dyn Retriever>,
) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
    let sources = search::catalog();
    let fallback = cfg.tools.live_default_source.trim();
    if !sources.iter().any(|s| s.key() == fallback) {
        anyhow::bail!(
            "tools.live_default_source is {fallback:?}, which is not a source the engine offers: {}",
            search::source_keys(&sources, false).join(", ")
        );
    }
    // A mismatch with the published registry is logged, not fatal.
    let published = queries.sources().await?;
    let mut unpublished = Vec::new();
    for source in &sources {
        let Some(served) = published.iter().find(|p| p.key == source.key()) else {
            unpublished.push(source.key());
            continue;
        };
        if let Err(difference) = search::conforms(source.as_ref(), served) {
            tracing::warn!(
                source = source.key(),
                %difference,
                "the tool and the scraper registry disagree; restart `just scraper serve` if it runs older code"
            );
        }
    }
    // One line for every unpublished source.
    if !unpublished.is_empty() {
        tracing::warn!(
            count = unpublished.len(),
            sources = %unpublished.join(", "),
            "the scraper has published no registry for these sources; run `just scraper serve`"
        );
    }
    let stored: Arc<dyn Tool> = Arc::new(StoredSearch::new(
        &sources,
        retriever,
        cfg.retrieval.top_k,
        &StoredWording {
            tool: cfg.tools.knowledge_description.clone(),
            query: cfg.tools.query_description.clone(),
            source: cfg.tools.source_description.clone(),
            empty: cfg.tools.nothing_stored.clone(),
        },
    ));
    let live: Arc<dyn Tool> = Arc::new(LiveSearch::new(
        sources,
        queries,
        fallback,
        cfg.prompt.utc_offset_hours,
        &LiveWording {
            tool: cfg.tools.live_description.clone(),
            query: cfg.tools.query_description.clone(),
            source: cfg.tools.source_description.clone(),
        },
        cfg.query.timeout_secs,
    ));
    tracing::info!(fallback, "search tools registered");
    Ok(vec![stored, live])
}

/// The registry and queue the scraper serves, behind the shared cache when one is configured.
pub(super) async fn source_queries(
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

/// The sandbox, when it is enabled and its runtime answers. Starts the session sweeper.
pub(super) async fn sandbox(cfg: &Config) -> anyhow::Result<Option<Arc<ContainerSandbox>>> {
    if !cfg.sandbox.enabled {
        return Ok(None);
    }
    let sandbox = ContainerSandbox::new(SandboxLimits::from(&cfg.sandbox));
    if let Err(error) = sandbox.probe().await {
        if cfg.sandbox.required {
            anyhow::bail!(
                "sandbox.enabled is on but the container runtime is unreachable: {error}. \
                 Give the engine a runtime, or set sandbox.required = false to run without it."
            );
        }
        tracing::warn!(%error, "the container runtime is unreachable; run_sandbox is not offered");
        return Ok(None);
    }
    // The sweep interval is the idle budget: a session lives at most twice it.
    let every = Duration::from_secs(cfg.sandbox.session_idle_secs.max(1));
    // Session containers a previous engine left running are nobody's to reap but this one's.
    sandbox.remove_orphans().await;
    tokio::spawn(reap_sessions(sandbox.clone(), every));
    Ok(Some(Arc::new(sandbox)))
}
