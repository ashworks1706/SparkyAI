//! The HTTP surface: the state each group of routes reads, and the router over them.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::core::config::Config;
use crate::core::traits::memory::profile::ProfileGraph;
use crate::core::traits::oauth::OAuthStore;
use crate::core::traits::safety::confirmation::ConfirmationStore;
use crate::core::traits::tools::sandbox::Sandbox;
use crate::core::types::tools::oauth::OAuthError;
use crate::routes::Limits;
use crate::routes::chat::ChatState;
use crate::routes::health::HealthState;
use crate::routes::oauth::OAuthState;
use crate::routes::profile::ProfileState;
use crate::routes::rate_limit::RateLimiter;
use crate::runtime::harness::agent::Agent;
use crate::runtime::tools::oauth::WebOAuthClient;
use crate::runtime::tools::sandbox::ContainerSandbox;
use crate::stores::oauth::PgOAuth;
use crate::stores::postgres::PgConversations;

/// What the chat routes need, gathered from the settings that describe it.
pub(super) fn chat_state(
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
        max_files: cfg.agent.max_files,
        max_file_bytes: cfg.agent.max_file_bytes,
        turns: Arc::new(tokio::sync::Semaphore::new(cfg.agent.max_turns.max(1))),
        turn_wait: Duration::from_secs(cfg.agent.turn_queue_wait_secs),
        attachments_only_input: cfg.prompt.attachments_only_input.clone(),
        service_token: cfg.engine.service_token.clone(),
        rate_limit: RateLimiter::new(cfg.http.rate_limit_per_min),
    }
}

/// What the profile routes need, gathered from the settings that describe it.
pub(super) fn profile_state(
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

/// What the sandbox routes read and drive.
pub(super) fn sandbox_state(
    cfg: &Config,
    sandbox: Option<Arc<ContainerSandbox>>,
) -> crate::routes::sandbox::SandboxState {
    crate::routes::sandbox::SandboxState {
        sandbox: sandbox.map(|s| s as Arc<dyn Sandbox>),
        service_token: cfg.engine.service_token.clone(),
    }
}

/// The grant store, a web OAuth client per enabled provider, and the state the OAuth routes read.
pub(super) type OAuthWiring = (
    Arc<dyn OAuthStore>,
    HashMap<String, Arc<WebOAuthClient>>,
    OAuthState,
);

/// Builds the web OAuth client of one provider.
type OAuthBuild<'a> = Box<dyn FnOnce() -> Result<WebOAuthClient, OAuthError> + 'a>;

/// Builds the grant store, the per-provider OAuth clients, and the OAuth route state.
pub(super) fn oauth_wiring(
    cfg: &Config,
    pool: &sqlx::postgres::PgPool,
) -> anyhow::Result<OAuthWiring> {
    let store: Arc<dyn OAuthStore> = Arc::new(PgOAuth::new(pool.clone()));
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
    let state = OAuthState {
        store: store.clone(),
        providers: providers.clone(),
        service_token: cfg.engine.service_token.clone(),
        // A login link stays valid for ten minutes.
        state_ttl: Duration::from_mins(10),
    };
    Ok((store, providers, state))
}

/// The whole HTTP surface, with the state each group of routes reads.
pub(super) fn http_router(
    cfg: &Config,
    state: ChatState,
    profile: ProfileState,
    oauth: OAuthState,
    pool: sqlx::postgres::PgPool,
    sandbox: Option<Arc<ContainerSandbox>>,
) -> axum::Router {
    let health = HealthState {
        pool,
        model_base_url: cfg.model.base_url.clone(),
    };
    let limits = Limits {
        max_body_bytes: cfg.http.max_body_bytes,
        concurrency: cfg.http.concurrency_limit,
    };
    crate::routes::router(
        state,
        health,
        profile,
        sandbox_state(cfg, sandbox),
        oauth,
        limits,
        &cfg.http.cors_origins,
    )
}
