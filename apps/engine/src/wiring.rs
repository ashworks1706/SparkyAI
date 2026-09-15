//! Construct concrete adapters, hand them to the harness, build the router, serve.

use std::sync::Arc;
use std::time::Duration;

use crate::core::config::Config;
use crate::core::traits::conversation::compaction::Compactor;
use crate::core::traits::knowledge::admission::Admission;
use crate::core::traits::knowledge::cache::QueryCache;
use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::traits::knowledge::retrieval::{Embedder, Retriever};
use crate::core::traits::knowledge::route::Router;
use crate::core::traits::knowledge::skills::SkillStore;
use crate::core::traits::memory::detector::FactDetector;
use crate::core::traits::memory::profile::ProfileGraph;
use crate::core::traits::model::ModelProvider;
use crate::core::traits::safety::confirmation::ConfirmationStore;
use crate::core::traits::safety::guardrail::Guardrail;
use crate::core::traits::tools::Tool;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::AgentConfig;
use crate::core::types::model::tokens::estimate;
use crate::routes::Limits;
use crate::routes::chat::ChatState;
use crate::routes::health::HealthState;
use crate::routes::profile::ProfileState;
use crate::routes::rate_limit::RateLimiter;
use crate::runtime::harness::agent::prompt::capability;
use crate::runtime::harness::agent::task::{Task, TaskConfig};
use crate::runtime::harness::agent::{Agent, AgentDeps, PromptText};
use crate::runtime::harness::compact::{self, ChatCompactor};
use crate::runtime::harness::knowledge::admit::AdmittedQueries;
use crate::runtime::harness::knowledge::cache::{CacheRules, CachedQueries};
use crate::runtime::harness::knowledge::route::{RuleRouter, Rules as RouterRules};
use crate::runtime::harness::memory::detect::{RuleDetector, Rules as DetectorRules};
use crate::runtime::harness::memory::profile::{self, GraphAgent, ProfileWriter, Reconciler};
use crate::runtime::harness::safety::guardrail::{RuleGuardrail, Rules};
use crate::runtime::harness::safety::policy::RiskPolicy;
use crate::runtime::harness::tools::ToolSet;
use crate::runtime::harness::trace::{Fanout, JsonlSink, NullSink};
use crate::runtime::model::limit::Limited;
use crate::runtime::model::props;
use crate::runtime::model::rig_openai::{self, RigChat, RigEmbedder};
use crate::runtime::tools::knowledge::search;
use crate::runtime::tools::knowledge::search::live::{LiveSearch, Wording as LiveWording};
use crate::runtime::tools::knowledge::search::stored::{StoredSearch, Wording as StoredWording};
use crate::runtime::tools::knowledge::skills::GetSkillTool;
use crate::runtime::tools::mcp::{self, McpLimits};
use crate::runtime::tools::sandbox::{ContainerSandbox, Limits as SandboxLimits, SandboxTool};
use crate::stores::knowledge::cache::{self as redis_cache, RedisAdmission, RedisQueryCache};
use crate::stores::knowledge::skills::PgSkills;
use crate::stores::memory::profile::PgProfileGraph;
use crate::stores::postgres::{
    self, PgConfirmations, PgConversations, PgMemory, PgRetriever, PgSourceQueries, RetrievalTuning,
};

/// Default system prompt, used when neither prompt.system_file nor system is set; hashed in traces.
pub const SYSTEM_PROMPT: &str = r#"You are Sparky, the assistant of the ASU AI Society, and you
answer students in Discord. Your subject is Arizona State University: courses, clubs, events,
library and dining hours, transit, deadlines, campus services, and the society itself.

## How you answer
- Ground every claim in the knowledge base results in this prompt or in tool output from this
  turn. You hold no reliable memory of ASU facts, so never answer one from memory of the web.
- Cite the bracketed number of each result you used, like [2]. Cite only numbers that appear in
  this prompt, and never invent a number, a URL, or a date.
- Lead with the answer, then the detail behind it. Two or three sentences is usually right. Use
  a short bullet list for hours, steps, or several items, and Discord markdown, never headings.
- When the results do not cover the question, say so plainly and name where to look. An honest
  gap costs a student less than a confident wrong answer.
- Ask one clarifying question only when the question has two readings that lead somewhere
  different. Otherwise answer.
- Match the label the question asks for. A row of hours carries one value per day, and the
  first value in the row answers a different question.

## Using your capabilities
The knowledge base results were retrieved for you before you were called. Read them first.
- You have two searches. search_knowledge reads the stored copies of ASU pages. search_live
  fetches a source, or the open web, as it is right now.
- Call search_live when the answer has to be current: hours today, open seats, shuttle times,
  events, news, scores. Nothing in this prompt can answer one of those.
- Call search_knowledge when the results here do not answer or are missing a detail.
- Both take a query and, optionally, a source. Leave source out unless you already know which
  source holds the answer.

The query carries the whole request. The tool reads nothing else from the conversation, so
write the subject in full, in keywords, every time. Never send a pronoun, a single bare word,
or a word you only have from an earlier message.
- "any AI clubs", nothing relevant in the results: search_knowledge with query artificial
  intelligence student club, source clubs.
- "does CSE 310 have open seats this fall": search_live with query CSE 310 open seats, source
  courses.
- "when is the next shuttle to Poly": search_live with query polytechnic-tempe shuttle next
  departure, source shuttles.
- "where is BYENG": search_live with query BYENG building, source campus_map.
- "what was the score of the ASU game last night": search_live with query ASU football score
  last night, no source.
- "when does hayden close tonight", library hours in the results: answer from them, cite them,
  call nothing.
- "what is a transformer": general knowledge, no ASU fact in it, answer directly and briefly.

An empty result means the answer is not held. Say so, or search once more with the subject
named differently. An action that needs approval waits for the user to press the button; never
say you did something you have only proposed.

## Never
- Never guess a date, room, price, deadline, policy, or person.
- Never repeat a search whose query you would write the same way twice.
- Never quote a result you were not given.
- Never tell the user to check the official site when you have just cited it.
- Never use run_sandbox to fetch a page or reach a site: it has no network. Use a search tool
  for the topic."#;

/// Serves until shutdown.
pub async fn serve(cfg: Config) -> anyhow::Result<()> {
    let chat_client = rig_openai::client(&cfg.model.base_url, &cfg.model.api_key)
        .map_err(|e| anyhow::anyhow!("model client: {e}"))?;
    let chat = Arc::new(RigChat::new(
        chat_client,
        &cfg.model.name,
        cfg.model.additional_params()?,
    ));
    fits_the_slot(&cfg).await?;
    let model: Arc<dyn ModelProvider> = if cfg.agent.model_slots == 0 {
        chat
    } else {
        Arc::new(Limited::new(
            chat,
            cfg.agent.model_slots,
            Duration::from_secs(cfg.agent.model_queue_wait_secs),
        ))
    };

    let trace: Arc<dyn TraceSink> =
        Arc::new(Fanout::new(trace_sink(&cfg)?).with_style(cfg.agent.progress_style()));

    let embed_client = rig_openai::client(&cfg.embedding.base_url, &cfg.embedding.api_key)
        .map_err(|e| anyhow::anyhow!("embedding client: {e}"))?;
    let embedder = Arc::new(RigEmbedder::new(
        embed_client,
        &cfg.embedding.name,
        usize::try_from(cfg.embedding.dim)?,
    ));

    // Every configured dependency must be reachable at boot.
    let pool = postgres::connect(
        &cfg.postgres.url,
        cfg.postgres.max_connections,
        Duration::from_secs(cfg.postgres.acquire_timeout_secs),
    )
    .await
    .map_err(|e| anyhow::anyhow!("postgres: {e}"))?;
    let retriever: Arc<dyn Retriever> = Arc::new(PgRetriever::new(
        pool.clone(),
        Arc::clone(&embedder) as Arc<dyn Embedder>,
        RetrievalTuning::from(&cfg.retrieval),
    ));
    let conversations = Arc::new(PgConversations::new(pool.clone()));
    let memory = Arc::new(PgMemory::new(pool.clone()));
    let confirmations: Arc<dyn ConfirmationStore> = Arc::new(PgConfirmations::new(pool.clone()));

    let queries = source_queries(&cfg, &pool, Arc::clone(&trace)).await?;
    let (tools, mcp_names) = build_tools(
        &cfg,
        &pool,
        queries,
        Arc::clone(&retriever) as Arc<dyn Retriever>,
    )
    .await?;
    let capabilities = capability::render(&capability::from_definitions(
        &tools.definitions(),
        &mcp_names,
    ));
    fits_the_prompt(&cfg, &tools, &capabilities)?;

    let agent_cfg = agent_config(&cfg);

    let profile_graph = profile_graph(&cfg, &pool, &embedder);
    let deps = AgentDeps {
        profile: profile_writer(&cfg, &model, profile_graph.clone()),
        profile_graph: profile_graph.clone(),
        compactor: compactor(&cfg, &model),
        guardrail: cfg.guardrail.enabled.then(|| {
            Arc::new(RuleGuardrail::new(Rules::from(&cfg.guardrail))) as Arc<dyn Guardrail>
        }),
        model,
        tools,
        policy: Arc::new(RiskPolicy::from(&cfg.policy)),
        trace,
        retriever: Some(retriever),
        router: router(&cfg),
        conversations: Some(conversations.clone()),
        memory: Some(memory),
        confirmations: Some(confirmations.clone()),
    };
    let system_prompt = cfg.system_prompt(SYSTEM_PROMPT)?;
    let agent = Agent::new(deps, agent_cfg, system_prompt)
        .with_prompt_text(PromptText::from(&cfg.prompt))
        .with_capabilities(capabilities);

    let state = chat_state(&cfg, agent, conversations, confirmations);
    let health = HealthState {
        pool,
        model_base_url: cfg.model.base_url.clone(),
    };
    let limits = Limits {
        max_body_bytes: cfg.http.max_body_bytes,
        concurrency: cfg.http.concurrency_limit,
    };

    let listener = tokio::net::TcpListener::bind(&cfg.app.http_addr).await?;
    tracing::info!(addr = %cfg.app.http_addr, "listening");
    let profile = profile_state(&cfg, profile_graph, state.rate_limit.clone());
    let router = crate::routes::router(state, health, profile, limits, &cfg.http.cors_origins);
    until_shutdown(
        listener,
        router,
        Duration::from_secs(cfg.http.shutdown_grace_secs),
    )
    .await
}

/// Serves until the shutdown signal, then gives in-flight requests grace to finish.
async fn until_shutdown(
    listener: tokio::net::TcpListener,
    router: axum::Router,
    grace: Duration,
) -> anyhow::Result<()> {
    let (signalled, wait) = tokio::sync::oneshot::channel();
    let server = axum::serve(listener, router).with_graceful_shutdown(async move {
        shutdown().await;
        // The receiver is gone only when the server has already stopped.
        let _ = signalled.send(());
    });
    // The grace period bounds how long in-flight requests may finish.
    tokio::select! {
        result = server => result?,
        () = expire(wait, grace) => {
            tracing::warn!(grace_secs = grace.as_secs(), "grace expired; dropping in-flight requests");
        }
    }
    Ok(())
}

/// Resolves grace after the shutdown signal, and never if it does not arrive.
async fn expire(wait: tokio::sync::oneshot::Receiver<()>, grace: Duration) {
    if wait.await.is_err() {
        std::future::pending::<()>().await;
    }
    tokio::time::sleep(grace).await;
}

/// What the chat routes need, gathered from the settings that describe it.
fn chat_state(
    cfg: &Config,
    agent: Agent,
    conversations: Arc<PgConversations>,
    confirmations: Arc<dyn ConfirmationStore>,
) -> ChatState {
    ChatState {
        confirmations: Some(confirmations),
        agent,
        conversations: Some(conversations),
        request_budget: Duration::from_secs(cfg.agent.request_timeout_secs),
        default_tenant: cfg.discord.guild_id.to_string(),
        max_images: cfg.agent.max_images,
        service_token: cfg.engine.service_token.clone(),
        rate_limit: RateLimiter::new(cfg.http.rate_limit_per_min),
    }
}

/// What the profile routes need, gathered from the settings that describe it.
fn profile_state(
    cfg: &Config,
    graph: Option<Arc<dyn ProfileGraph>>,
    rate_limit: RateLimiter,
) -> ProfileState {
    ProfileState {
        graph,
        list_limit: cfg.profile.list_limit,
        request_budget: Duration::from_secs(cfg.profile.request_timeout_secs),
        rate_limit,
        default_tenant: cfg.discord.guild_id.to_string(),
        service_token: cfg.engine.service_token.clone(),
    }
}

/// The profile graph, when profile recording is on.
fn profile_graph(
    cfg: &Config,
    pool: &sqlx::PgPool,
    embedder: &Arc<RigEmbedder>,
) -> Option<Arc<dyn ProfileGraph>> {
    cfg.profile.enabled.then(|| {
        Arc::new(PgProfileGraph::new(
            pool.clone(),
            Arc::clone(embedder) as Arc<dyn Embedder>,
        )) as Arc<dyn ProfileGraph>
    })
}

/// The classifier and the graph agent, when profile recording is on.
fn profile_writer(
    cfg: &Config,
    model: &Arc<dyn ModelProvider>,
    graph: Option<Arc<dyn ProfileGraph>>,
) -> Option<Arc<ProfileWriter>> {
    let graph = graph?;
    let budget = Duration::from_secs(cfg.profile.timeout_secs);
    let instructions = |set: Option<&String>, fallback: &'static str| {
        set.map(String::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(fallback)
            .to_owned()
    };
    // The gate is rule based.
    let detector: Arc<dyn FactDetector> = Arc::new(RuleDetector::new(DetectorRules::from(
        &cfg.profile.detector,
    )));
    let agent = GraphAgent::new(Task::new(
        Arc::clone(model),
        "profile.extract",
        instructions(
            cfg.profile.graph_instructions.as_ref(),
            profile::GRAPH_INSTRUCTIONS,
        ),
        TaskConfig {
            max_tokens: cfg.profile.max_tokens,
            temperature: 0.0,
            timeout: budget,
            ..task_config(cfg)
        },
    ));
    let reconciler = cfg.profile.reconcile.then(|| {
        Reconciler::new(Task::new(
            Arc::clone(model),
            "profile.reconcile",
            instructions(
                cfg.profile.reconcile_instructions.as_ref(),
                profile::RECONCILE_INSTRUCTIONS,
            ),
            TaskConfig {
                // A list of numbers, or the word none.
                max_tokens: 32,
                temperature: 0.0,
                timeout: budget,
                ..task_config(cfg)
            },
        ))
    });
    Some(Arc::new(ProfileWriter::new(
        detector,
        agent,
        reconciler,
        graph,
        budget,
        cfg.profile.min_confidence,
    )))
}

/// The chat agent, when compaction is on. Shares the model the loop calls.
fn compactor(cfg: &Config, model: &Arc<dyn ModelProvider>) -> Option<Arc<dyn Compactor>> {
    if !cfg.compaction.enabled {
        return None;
    }
    let instructions = cfg
        .compaction
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(compact::INSTRUCTIONS);
    let task = Task::new(
        Arc::clone(model),
        "compaction",
        instructions,
        TaskConfig {
            max_tokens: cfg.compaction.max_tokens,
            temperature: cfg.compaction.temperature,
            timeout: Duration::from_secs(cfg.compaction.timeout_secs),
            ..task_config(cfg)
        },
    );
    Some(Arc::new(ChatCompactor::new(task)))
}

/// Where traces are recorded, or a sink dropping them if off; old traces are pruned here once.
fn trace_sink(cfg: &Config) -> anyhow::Result<Arc<dyn TraceSink>> {
    if !cfg.trace.enabled {
        tracing::info!("jsonl traces are off");
        return Ok(Arc::new(NullSink));
    }
    let sink = JsonlSink::new(&cfg.trace.dir, cfg.trace.max_file_bytes)?;
    if cfg.trace.retention_hours > 0 {
        let older_than = Duration::from_secs(cfg.trace.retention_hours * 3_600);
        match sink.prune(older_than) {
            Ok(removed) if removed > 0 => {
                tracing::info!(removed, dir = %cfg.trace.dir, "pruned old traces");
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(error = %e, dir = %cfg.trace.dir, "trace prune failed"),
        }
    }
    Ok(Arc::new(sink))
}

/// A task config carrying the span settings, with the call settings at their defaults.
fn task_config(cfg: &Config) -> TaskConfig {
    TaskConfig {
        provider_name: cfg.telemetry.provider_name.as_str().into(),
        model_name: cfg.model.name.as_str().into(),
        max_span_value_chars: cfg.agent.max_span_value_chars,
        ..TaskConfig::default()
    }
}

/// Fails boot if the biggest prompt plus reply cannot fit one slot; skipped if context unreported.
async fn fits_the_slot(cfg: &Config) -> anyhow::Result<()> {
    let slot = match props::slot_context(&cfg.model.base_url, &cfg.model.api_key).await {
        Ok(slot) => u64::from(slot),
        Err(error) => {
            tracing::warn!(%error, "could not read the chat server context; the prompt budget is not checked against it");
            return Ok(());
        }
    };
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let prompt = (cfg.agent.prompt_budget_tokens as f64
        * (1.0 + cfg.agent.prompt_estimate_headroom))
        .ceil() as u64;
    let plain = prompt + u64::from(cfg.model.max_tokens_without_thinking);
    if plain > slot {
        anyhow::bail!(
            "one chat server slot holds {slot} tokens, but a prompt of up to {prompt} tokens \
             (agent.prompt_budget_tokens with agent.prompt_estimate_headroom) and \
             model.max_tokens_without_thinking need {plain}; raise SPARKY_CHAT_CTX or lower \
             SPARKY_CHAT_PARALLEL, or lower those settings"
        );
    }
    let thinking = prompt + u64::from(cfg.model.max_tokens);
    if thinking > slot {
        tracing::warn!(
            slot,
            needed = thinking,
            "a call that thinks can run out of room before it answers"
        );
    }
    Ok(())
}

/// Fails boot if tools crowd the prompt: capabilities must fit its budget, schemas leave half free.
fn fits_the_prompt(cfg: &Config, tools: &ToolSet, capabilities: &str) -> anyhow::Result<()> {
    let cpt = cfg.agent.chars_per_token;
    let listed = estimate(capabilities, cpt);
    if listed > cfg.agent.capabilities_budget_tokens {
        anyhow::bail!(
            "the capabilities section needs about {listed} tokens but \
             agent.capabilities_budget_tokens is {}; raise it or disable tools",
            cfg.agent.capabilities_budget_tokens
        );
    }
    let schemas = tools.estimated_tokens(cpt);
    if schemas > cfg.agent.prompt_budget_tokens / 2 {
        anyhow::bail!(
            "the tool schemas need about {schemas} tokens, over half of \
             agent.prompt_budget_tokens ({}); raise it or disable tools",
            cfg.agent.prompt_budget_tokens
        );
    }
    Ok(())
}

/// The loop limits and budgets, gathered from the sections that own them.
fn agent_config(cfg: &Config) -> AgentConfig {
    AgentConfig {
        provider_name: cfg.telemetry.provider_name.as_str().into(),
        model_name: cfg.model.name.as_str().into(),
        max_steps: cfg.agent.max_steps,
        max_model_retries: cfg.agent.max_model_retries,
        tool_timeout: Duration::from_secs(cfg.agent.tool_timeout_secs),
        confirmation_ttl: Duration::from_secs(cfg.agent.confirmation_ttl_secs),
        max_tokens: cfg.model.max_tokens,
        max_tokens_without_thinking: cfg.model.max_tokens_without_thinking,
        temperature: cfg.agent.temperature,
        retrieval_top_k: cfg.retrieval.top_k,
        history_turns: cfg.agent.history_turns,
        memory_recall_limit: cfg.agent.memory_recall_limit,
        recall_in_public: cfg.agent.recall_in_public,
        retry_base_ms: cfg.agent.retry_base_ms,
        retry_cap_ms: cfg.agent.retry_cap_ms,
        max_span_value_chars: cfg.agent.max_span_value_chars,
        usd_per_m_prompt: cfg.model.usd_per_m_prompt,
        usd_per_m_completion: cfg.model.usd_per_m_completion,
        budget: cfg.agent.budget(),
        thinking: cfg.agent.thinking.clone(),
        stream: cfg.agent.stream,
        stream_block_chars: cfg.agent.stream_block_chars,
    }
}

/// The gate retrieval passes, when it is on. Off retrieves for every turn.
fn router(cfg: &Config) -> Option<Arc<dyn Router>> {
    cfg.retrieval.router.enabled.then(|| {
        Arc::new(RuleRouter::new(RouterRules::from(&cfg.retrieval.router))) as Arc<dyn Router>
    })
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

    // The cap goes on first so the cache sits above it: a reused answer and a request that
    // waited on another cost the database nothing and take no slot.
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
    }
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
    for source in &sources {
        let Some(served) = published.iter().find(|p| p.key == source.key()) else {
            tracing::warn!(
                source = source.key(),
                "the scraper has not published this source; run `just scraper serve`"
            );
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

/// Every tool the model may call, with tools.disabled removed at registration.
async fn build_tools(
    cfg: &Config,
    pool: &sqlx::PgPool,
    queries: Arc<dyn SourceQueries>,
    retriever: Arc<dyn Retriever>,
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
    if cfg.tools.get_skill {
        // get_skill registers only when reviewed skills exist.
        let skills = PgSkills::new(pool.clone());
        let offered = skills.list().await?;
        if offered.is_empty() {
            tracing::info!("no reviewed skills; get_skill is not offered");
        } else {
            let tool: Arc<dyn Tool> = Arc::new(GetSkillTool::new(Arc::new(skills), &offered));
            if !disabled(&tool.definition().name) {
                tracing::info!(count = offered.len(), "skills registered");
                tools = tools.with(tool);
            }
        }
    }
    if cfg.sandbox.enabled {
        let sandbox = Arc::new(ContainerSandbox::new(SandboxLimits::from(&cfg.sandbox)));
        let tool: Arc<dyn Tool> = Arc::new(SandboxTool::new(sandbox, cfg.sandbox.risk));
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
        let remote = mcp::connect(&server.url, &server.tools, &limits)
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
    tracing::info!(tools = ?tools, "tool set");
    Ok((tools, mcp_names))
}

/// Resolves on Ctrl-C or SIGTERM.
async fn shutdown() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "could not listen for ctrl-c");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(error) => {
                tracing::error!(%error, "could not listen for SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
    tracing::info!("shutting down");
}
