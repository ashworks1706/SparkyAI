//! Construct concrete adapters, hand them to the harness, build the router, serve.

use std::sync::Arc;
use std::time::Duration;

use crate::agent::harness::agent::{Agent, AgentDeps};
use crate::agent::harness::policy::RiskPolicy;
use crate::agent::harness::tool::ToolSet;
use crate::agent::harness::trace::JsonlSink;
use crate::agent::model::rerank::HttpReranker;
use crate::agent::model::rig_openai::{self, RigChat, RigEmbedder};
use crate::agent::tools::discord_ops::PostAnnouncement;
use crate::agent::tools::public_search::PublicSearch;
use crate::core::config::Config;
use crate::core::traits::trace::TraceSink;
use crate::core::types::agent::AgentConfig;
use crate::core::types::assemble::Budget;
use crate::routes::chat::ChatState;
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
    let model = Arc::new(RigChat::new(chat_client, &cfg.model.name));

    let trace: Arc<dyn TraceSink> = Arc::new(JsonlSink::new(&cfg.agent.trace_dir)?);

    let embed_client = rig_openai::client(&cfg.embedding.base_url, &cfg.embedding.api_key)
        .map_err(|e| anyhow::anyhow!("embedding client: {e}"))?;
    let embedder = Arc::new(RigEmbedder::new(
        embed_client,
        &cfg.embedding.name,
        usize::try_from(cfg.embedding.dim)?,
    ));
    let reranker = Arc::new(HttpReranker::new(
        &cfg.reranker.base_url,
        cfg.reranker.api_key.clone(),
        &cfg.reranker.name,
    )?);

    // Stores are optional at boot so the harness can run against a model alone.
    let pool = match postgres::connect(&cfg.postgres.url, cfg.postgres.max_connections).await {
        Ok(pool) => Some(pool),
        Err(e) => {
            tracing::warn!(error = %e, "postgres unavailable; running without retrieval, history, or memory");
            None
        }
    };
    let retriever = pool
        .clone()
        .map(|p| Arc::new(PgRetriever::new(p, embedder, Some(reranker))));
    let conversations = pool.clone().map(|p| Arc::new(PgConversations::new(p)));
    let memory = pool.map(|p| Arc::new(PgMemory::new(p)));

    let mut tools = ToolSet::new().with(Arc::new(PostAnnouncement));
    if let Some(r) = &retriever {
        tools = tools.with(Arc::new(PublicSearch::new(
            r.clone(),
            cfg.agent.retrieval_top_k,
        )));
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
        policy: Arc::new(RiskPolicy::new(Some(cfg.discord.mod_role.clone()))),
        trace,
        retriever: retriever.map(|r| r as _),
        conversations: conversations.clone().map(|c| c as _),
        memory: memory.map(|m| m as _),
    };
    let agent = Agent::new(deps, agent_cfg, SYSTEM_PROMPT);

    let state = ChatState {
        agent,
        conversations: conversations.map(|c| c as _),
        request_budget: Duration::from_secs(cfg.agent.request_timeout_secs),
        default_tenant: cfg.discord.guild_id.to_string(),
    };

    let listener = tokio::net::TcpListener::bind(&cfg.app.http_addr).await?;
    tracing::info!(addr = %cfg.app.http_addr, "listening");
    axum::serve(listener, crate::routes::router(state)).await?;
    Ok(())
}
