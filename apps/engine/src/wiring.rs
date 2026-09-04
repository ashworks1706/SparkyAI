//! Construct concrete adapters, hand them to the harness, build the router, serve.

use std::sync::Arc;
use std::time::Duration;

use crate::agent::harness::agent::{Agent, AgentDeps};
use crate::agent::harness::policy::RiskPolicy;
use crate::agent::harness::tool::ToolSet;
use crate::agent::harness::trace::JsonlSink;
use crate::agent::model::rig_openai::{self, RigChat, RigEmbedder};
use crate::agent::tools::mcp;
use crate::agent::tools::public_search::PublicSearch;
use crate::core::config::Config;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::AgentConfig;
use crate::core::types::assemble::Budget;
use crate::routes::chat::ChatState;
use crate::routes::health::HealthState;
use crate::stores::postgres::{self, PgConversations, PgMemory, PgRetriever};

/// Default system prompt. Versioned by content; changes show up in traces via the prompt hash.
const SYSTEM_PROMPT: &str = "You are Sparky, the ASU AI Society's assistant on Discord. \
Answer from the evidence you are given or from tools; never from memory of the web. \
When you cite, use the bracketed evidence numbers. If the evidence does not answer the \
question, say so plainly and suggest where the user might look. Be brief.";

/// Serves until shutdown.
pub async fn serve(cfg: Config) -> anyhow::Result<()> {
    let chat_client = rig_openai::client(&cfg.model.base_url, &cfg.model.api_key)
        .map_err(|e| anyhow::anyhow!("model client: {e}"))?;
    let model = Arc::new(RigChat::new(
        chat_client,
        &cfg.model.name,
        cfg.model.thinking,
    ));

    let trace: Arc<dyn TraceSink> = Arc::new(JsonlSink::new(&cfg.agent.trace_dir)?);

    let embed_client = rig_openai::client(&cfg.embedding.base_url, &cfg.embedding.api_key)
        .map_err(|e| anyhow::anyhow!("embedding client: {e}"))?;
    let embedder = Arc::new(RigEmbedder::new(
        embed_client,
        &cfg.embedding.name,
        usize::try_from(cfg.embedding.dim)?,
    ));

    // Every configured dependency must be reachable at boot. Nothing degrades silently.
    let pool = postgres::connect(&cfg.postgres.url, cfg.postgres.max_connections)
        .await
        .map_err(|e| anyhow::anyhow!("postgres: {e}"))?;
    let retriever = Arc::new(PgRetriever::new(pool.clone(), embedder));
    let conversations = Arc::new(PgConversations::new(pool.clone()));
    let memory = Arc::new(PgMemory::new(pool.clone()));

    let mut tools = ToolSet::new().with(Arc::new(PublicSearch::new(
        retriever.clone(),
        cfg.agent.retrieval_top_k,
    )));
    if let Some(url) = cfg
        .mcp
        .playwright_url
        .as_deref()
        .filter(|u| !u.trim().is_empty())
    {
        let remote = mcp::connect(url, &cfg.mcp.playwright_tools, cfg.mcp.required_props_only)
            .await
            .map_err(|e| anyhow::anyhow!("playwright mcp at {url}: {e}"))?;
        tracing::info!(count = remote.len(), url, "playwright mcp tools registered");
        for tool in remote {
            tools = tools.with(tool);
        }
    }

    let agent_cfg = AgentConfig {
        max_steps: cfg.agent.max_steps,
        max_model_retries: cfg.agent.max_model_retries,
        tool_timeout: Duration::from_secs(cfg.agent.tool_timeout_secs),
        max_tokens: cfg.model.max_tokens,
        temperature: cfg.agent.temperature,
        retrieval_top_k: cfg.agent.retrieval_top_k,
        history_turns: cfg.agent.history_turns,
        usd_per_m_prompt: cfg.model.usd_per_m_prompt,
        usd_per_m_completion: cfg.model.usd_per_m_completion,
        budget: Budget {
            total: cfg.agent.prompt_budget_tokens,
            ..Budget::default()
        },
    };

    let deps = AgentDeps {
        model,
        tools,
        policy: Arc::new(RiskPolicy::new()),
        trace,
        retriever: Some(retriever),
        conversations: Some(conversations.clone()),
        memory: Some(memory),
    };
    let agent = Agent::new(deps, agent_cfg, SYSTEM_PROMPT);

    let state = ChatState {
        agent,
        conversations: Some(conversations),
        request_budget: Duration::from_secs(cfg.agent.request_timeout_secs),
        default_tenant: cfg.discord.guild_id.to_string(),
        service_token: cfg.engine.service_token.clone(),
    };
    let health = HealthState {
        pool,
        model_base_url: cfg.model.base_url.clone(),
    };

    let listener = tokio::net::TcpListener::bind(&cfg.app.http_addr).await?;
    tracing::info!(addr = %cfg.app.http_addr, "listening");
    axum::serve(listener, crate::routes::router(state, health)).await?;
    Ok(())
}
