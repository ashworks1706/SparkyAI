//! Construct concrete adapters, hand them to the harness, build the router, serve.

mod agent;
mod boot;
mod http;
mod prompt;
mod tools;

use std::sync::Arc;
use std::time::Duration;

use crate::core::config::Config;
use crate::core::traits::knowledge::retrieval::{Embedder, Retriever};
use crate::core::traits::model::ModelProvider;
use crate::core::traits::safety::confirmation::ConfirmationStore;
use crate::core::traits::tools::files::FileSource;
use crate::core::traits::tools::sandbox::Sandbox;
use crate::core::traits::trace::TraceSink;
use crate::runtime::harness::agent::prompt::capability;
use crate::runtime::harness::agent::{Agent, AgentDeps, PromptText};
use crate::runtime::harness::safety::policy::RiskPolicy;
use crate::runtime::harness::trace::Fanout;
use crate::runtime::model::limit::Limited;
use crate::runtime::model::rig_openai::{self, RigChat, RigEmbedder};
use crate::runtime::tools::files::HttpFiles;
use crate::stores::postgres::{
    self, PgConfirmations, PgConversations, PgMemory, PgRetriever, RetrievalTuning,
};

use self::agent::{agent_config, compactor, guardrail, profile_graph, profile_writer, trace_sink};
use self::boot::{fits_the_prompt, fits_the_slot};
use self::http::{chat_state, http_router, oauth_wiring, profile_state};
use self::prompt::SYSTEM_PROMPT;
use self::tools::{build_tools, sandbox, source_queries};

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
    let (oauth_store, oauth_providers, oauth_state) = oauth_wiring(&cfg, &pool)?;

    let queries = source_queries(&cfg, &pool, Arc::clone(&trace)).await?;
    let sandbox = sandbox(&cfg).await?;
    let (tools, mcp_names) = build_tools(
        &cfg,
        queries,
        Arc::clone(&retriever) as Arc<dyn Retriever>,
        sandbox.clone(),
        oauth_store,
        oauth_providers,
    )
    .await?;
    // Measured at boot with every tool offered, which is the largest the section ever gets.
    let definitions = tools.definitions();
    let capabilities = capability::render(&capability::from_definitions(&definitions, &mcp_names));
    fits_the_prompt(&cfg, &tools, &capabilities)?;
    // The tool names the guardrail keeps out of answers.
    let tool_names: Vec<String> = definitions.iter().map(|d| d.name.clone()).collect();

    let agent_cfg = agent_config(&cfg);

    let profile_graph = profile_graph(&cfg, &pool, &embedder);
    let deps = AgentDeps {
        profile: profile_writer(&cfg, &model, profile_graph.clone()),
        profile_graph: profile_graph.clone(),
        compactor: compactor(&cfg, &model),
        guardrail: guardrail(&cfg, &tool_names),
        model,
        tools,
        policy: Arc::new(RiskPolicy::from(&cfg.policy)),
        trace,
        conversations: Some(conversations.clone()),
        memory: Some(memory),
        confirmations: Some(confirmations.clone()),
        sandbox: sandbox.clone().map(|s| s as Arc<dyn Sandbox>),
        files: Some(Arc::new(
            HttpFiles::new(cfg.agent.file_hosts.clone(), cfg.agent.max_file_bytes)
                .map_err(|e| anyhow::anyhow!("attachments: {e}"))?,
        ) as Arc<dyn FileSource>),
    };
    let system_prompt = cfg.system_prompt(SYSTEM_PROMPT)?;
    let agent = Agent::new(deps, agent_cfg, system_prompt)
        .with_prompt_text(PromptText::from(&cfg.prompt))
        .with_mcp_names(mcp_names);

    let state = chat_state(&cfg, agent, conversations, confirmations);
    let profile = profile_state(&cfg, profile_graph, state.rate_limit.clone());
    let router = http_router(&cfg, state, profile, oauth_state, pool, sandbox);

    let listener = tokio::net::TcpListener::bind(&cfg.app.http_addr).await?;
    tracing::info!(addr = %cfg.app.http_addr, "listening");
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
