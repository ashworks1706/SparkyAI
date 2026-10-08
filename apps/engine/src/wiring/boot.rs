//! Boot checks that fail fast when the prompt cannot fit the model or the tools crowd it.

use crate::core::config::Config;
use crate::core::types::model::tokens::estimate;
use crate::runtime::harness::tools::ToolSet;
use crate::runtime::model::props;

/// Fails boot if the biggest prompt plus reply cannot fit one slot; skipped if context unreported.
pub(super) async fn fits_the_slot(cfg: &Config) -> anyhow::Result<()> {
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
pub(super) fn fits_the_prompt(
    cfg: &Config,
    tools: &ToolSet,
    capabilities: &str,
) -> anyhow::Result<()> {
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
