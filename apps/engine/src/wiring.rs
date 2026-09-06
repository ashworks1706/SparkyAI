//! Construct concrete adapters, hand them to the harness, build the router, serve.

use std::sync::Arc;
use std::time::Duration;

use crate::agent::harness::agent::{Agent, AgentDeps, PromptText};
use crate::agent::harness::policy::RiskPolicy;
use crate::agent::harness::tool::ToolSet;
use crate::agent::harness::trace::{Fanout, JsonlSink, NullSink};
use crate::agent::model::limit::Limited;
use crate::agent::model::rig_openai::{self, RigChat, RigEmbedder};
use crate::agent::tools::knowledge_search::KnowledgeSearch;
use crate::agent::tools::mcp::{self, McpLimits};
use crate::agent::tools::query_source::QuerySourceTool;
use crate::core::config::Config;
use crate::core::traits::confirmation::ConfirmationStore;
use crate::core::traits::model::ModelProvider;
use crate::core::traits::query::SourceQueries;
use crate::core::traits::tool::Tool;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::AgentConfig;
use crate::routes::Limits;
use crate::routes::chat::ChatState;
use crate::routes::health::HealthState;
use crate::routes::rate_limit::RateLimiter;
use crate::stores::postgres::{
    self, PgConfirmations, PgConversations, PgMemory, PgRetriever, PgSourceQueries, RetrievalTuning,
};

/// Default system prompt, used when neither prompt.system_file nor prompt.system is set.
/// Versioned by content; changes show up in traces via the prompt hash.
pub const SYSTEM_PROMPT: &str = "You are Sparky, the ASU AI Society's assistant on Discord. \
Answer from the evidence you are given or from tools; never from memory of the web. \
When you cite, use the bracketed evidence numbers. If the evidence does not answer the \
question, say so plainly and suggest where the user might look. Be brief.";

/// Serves until shutdown.
pub async fn serve(cfg: Config) -> anyhow::Result<()> {
    let chat_client = rig_openai::client(&cfg.model.base_url, &cfg.model.api_key)
        .map_err(|e| anyhow::anyhow!("model client: {e}"))?;
    let chat = Arc::new(RigChat::new(
        chat_client,
        &cfg.model.name,
        cfg.model.additional_params()?,
    ));
    let model: Arc<dyn ModelProvider> = if cfg.agent.model_slots == 0 {
        chat
    } else {
        Arc::new(Limited::new(
            chat,
            cfg.agent.model_slots,
            Duration::from_secs(cfg.agent.model_queue_wait_secs),
        ))
    };

    let trace: Arc<dyn TraceSink> = Arc::new(Fanout::new(trace_sink(&cfg)?));

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
    let retriever = Arc::new(PgRetriever::new(
        pool.clone(),
        embedder,
        RetrievalTuning::from(&cfg.retrieval),
    ));
    let conversations = Arc::new(PgConversations::new(pool.clone()));
    let memory = Arc::new(PgMemory::new(pool.clone()));
    let confirmations: Arc<dyn ConfirmationStore> = Arc::new(PgConfirmations::new(pool.clone()));

    let tools = build_tools(&cfg, retriever.clone(), source_queries(&cfg, &pool)).await?;

    let agent_cfg = agent_config(&cfg);

    let deps = AgentDeps {
        model,
        tools,
        policy: Arc::new(RiskPolicy::from(&cfg.policy)),
        trace,
        retriever: Some(retriever),
        conversations: Some(conversations.clone()),
        memory: Some(memory),
        confirmations: Some(confirmations.clone()),
    };
    let system_prompt = cfg.system_prompt(SYSTEM_PROMPT)?;
    let agent =
        Agent::new(deps, agent_cfg, system_prompt).with_prompt_text(PromptText::from(&cfg.prompt));

    let state = ChatState {
        confirmations: Some(confirmations),
        agent,
        conversations: Some(conversations),
        request_budget: Duration::from_secs(cfg.agent.request_timeout_secs),
        default_tenant: cfg.discord.guild_id.to_string(),
        service_token: cfg.engine.service_token.clone(),
        rate_limit: RateLimiter::new(cfg.http.rate_limit_per_min),
    };
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
    let router = crate::routes::router(state, health, limits, &cfg.http.cors_origins);
    let grace = Duration::from_secs(cfg.http.shutdown_grace_secs);
    let (signalled, wait) = tokio::sync::oneshot::channel();
    let server = axum::serve(listener, router).with_graceful_shutdown(async move {
        shutdown().await;
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

/// Where traces are recorded, or a sink that drops them when recording is off. Traces older
/// than the retention window are pruned once here.
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

/// The loop limits and budgets, gathered from the sections that own them.
fn agent_config(cfg: &Config) -> AgentConfig {
    AgentConfig {
        max_steps: cfg.agent.max_steps,
        max_model_retries: cfg.agent.max_model_retries,
        tool_timeout: Duration::from_secs(cfg.agent.tool_timeout_secs),
        confirmation_ttl: Duration::from_secs(cfg.agent.confirmation_ttl_secs),
        max_tokens: cfg.model.max_tokens,
        temperature: cfg.agent.temperature,
        retrieval_top_k: cfg.retrieval.top_k,
        history_turns: cfg.agent.history_turns,
        memory_recall_limit: cfg.agent.memory_recall_limit,
        retry_base_ms: cfg.agent.retry_base_ms,
        retry_cap_ms: cfg.agent.retry_cap_ms,
        max_span_value_chars: cfg.agent.max_span_value_chars,
        usd_per_m_prompt: cfg.model.usd_per_m_prompt,
        usd_per_m_completion: cfg.model.usd_per_m_completion,
        budget: cfg.agent.budget(),
    }
}

/// The registry and queue the scraper worker serves.
fn source_queries(cfg: &Config, pool: &sqlx::PgPool) -> Arc<dyn SourceQueries> {
    Arc::new(PgSourceQueries::new(
        pool.clone(),
        Duration::from_millis(cfg.query.poll_ms),
    ))
}

/// Every tool the model may call, with tools.disabled removed at registration.
async fn build_tools(
    cfg: &Config,
    retriever: Arc<PgRetriever>,
    queries: Arc<dyn SourceQueries>,
) -> anyhow::Result<ToolSet> {
    let disabled = |name: &str| cfg.tools.disabled.iter().any(|d| d == name);
    let mut tools = ToolSet::new();
    if cfg.tools.knowledge_search {
        let search: Arc<dyn Tool> = Arc::new(KnowledgeSearch::new(retriever, cfg.retrieval.top_k));
        if !disabled(&search.definition().name) {
            tools = tools.with(search);
        }
    }
    if cfg.tools.query_source {
        // An empty registry means the scraper has published no sources.
        let sources = queries.sources().await?;
        if sources.is_empty() {
            tracing::info!("no query sources registered; run `just worker` to publish them");
        } else {
            let tool: Arc<dyn Tool> = Arc::new(QuerySourceTool::new(
                queries,
                &sources,
                cfg.query.timeout_secs,
            ));
            if !disabled(&tool.definition().name) {
                tracing::info!(count = sources.len(), "query sources registered");
                tools = tools.with(tool);
            }
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
    Ok(tools)
}

/// Resolves on Ctrl-C or SIGTERM.
async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
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
