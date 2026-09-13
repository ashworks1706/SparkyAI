//! The /reset, /memory, and /forget commands, and the forget everything buttons.

use serenity::all::{
    CommandInteraction, ComponentInteraction, Context, CreateInteractionResponse,
    CreateInteractionResponseFollowup, CreateInteractionResponseMessage, EditInteractionResponse,
};
use tracing::Instrument;

use super::commands;
use super::{Handler, error_kind, event, option_str, tell};
use crate::core::types::{ForgetRequest, ProfileRequest, ResetRequest};
use crate::render::components::{self, CustomId};
use crate::render::reply;

/// Asked before erasing everything.
const FORGET_ALL_PROMPT: &str = "Forget everything I remember about you? This cannot be undone.";

impl Handler {
    /// Ends the conversation of the caller in this channel or thread.
    pub(super) async fn reset(&self, ctx: &Context, cmd: &CommandInteraction) {
        let Some(guild_id) = cmd.guild_id else {
            tell(ctx, cmd, "Use this in the server.").await;
            return;
        };
        if !defer_private(ctx, cmd).await {
            return;
        }
        let req = ResetRequest {
            user: cmd.user.id.to_string(),
            tenant: guild_id.to_string(),
            channel: cmd.channel_id.to_string(),
        };
        let span = tracing::info_span!(
            "discord.reset",
            "discord.command" = "reset",
            "posthog.distinct_id" = %cmd.user.id,
        );
        let text = match self.engine.reset(&req).instrument(span).await {
            Ok(done) => {
                tracing::info!(user = %cmd.user.id, ended = done.ended, "conversation reset");
                self.record(
                    event("discord_reset", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("ended", done.ended),
                );
                "Fresh start. Ask away.".to_owned()
            }
            Err(e) => {
                tracing::error!(error = %e, user = %cmd.user.id, "reset failed");
                self.record(
                    event("discord_error", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("stage", "reset")
                        .with("kind", error_kind(&e)),
                );
                reply::failure(&e)
            }
        };
        finish(ctx, cmd, vec![text]).await;
    }

    /// Lists what the engine remembers about the caller.
    pub(super) async fn memory(&self, ctx: &Context, cmd: &CommandInteraction) {
        let Some(guild_id) = cmd.guild_id else {
            tell(ctx, cmd, "Use this in the server.").await;
            return;
        };
        if !defer_private(ctx, cmd).await {
            return;
        }
        let req = ProfileRequest {
            user_id: cmd.user.id.to_string(),
            tenant_id: guild_id.to_string(),
        };
        let span = tracing::info_span!(
            "discord.memory",
            "discord.command" = "memory",
            "posthog.distinct_id" = %cmd.user.id,
        );
        let messages = match self.engine.profile_list(&req).instrument(span).await {
            Ok(profile) => {
                tracing::info!(
                    user = %cmd.user.id,
                    nodes = profile.nodes.len(),
                    relations = profile.relations.len(),
                    "profile listed"
                );
                self.record(
                    event("discord_memory", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("nodes", profile.nodes.len())
                        .with("relations", profile.relations.len()),
                );
                reply::render_profile(&profile, self.max_message_chars)
            }
            Err(e) => {
                tracing::warn!(error = %e, user = %cmd.user.id, "profile list failed");
                self.record(
                    event("discord_error", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("stage", "memory")
                        .with("kind", error_kind(&e)),
                );
                vec![reply::memory_failure(&e)]
            }
        };
        finish(ctx, cmd, messages).await;
    }

    /// Forgets one thing by label, or asks before forgetting everything.
    pub(super) async fn forget(&self, ctx: &Context, cmd: &CommandInteraction) {
        let Some(guild_id) = cmd.guild_id else {
            tell(ctx, cmd, "Use this in the server.").await;
            return;
        };
        let label = option_str(cmd, commands::LABEL)
            .map(|l| l.trim().to_owned())
            .filter(|l| !l.is_empty());
        let Some(label) = label else {
            let rows = components::forget_rows(cmd.user.id.get());
            let msg = CreateInteractionResponseMessage::new()
                .content(FORGET_ALL_PROMPT)
                .components(components::to_action_rows(&rows))
                .ephemeral(true);
            if let Err(e) = cmd
                .create_response(&ctx.http, CreateInteractionResponse::Message(msg))
                .await
            {
                tracing::warn!(error = %e, "forget prompt failed");
            }
            return;
        };
        if !defer_private(ctx, cmd).await {
            return;
        }
        let req = ForgetRequest {
            user_id: cmd.user.id.to_string(),
            tenant_id: guild_id.to_string(),
            label: Some(label),
        };
        let span = tracing::info_span!(
            "discord.forget",
            "discord.command" = "forget",
            "posthog.distinct_id" = %cmd.user.id,
            "sparky.everything" = false,
        );
        let text = match self.engine.forget(&req).instrument(span).await {
            Ok(done) => {
                tracing::info!(user = %cmd.user.id, removed = done.removed, "forgot by label");
                self.record(
                    event("discord_forget", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("removed", done.removed)
                        .with("everything", false),
                );
                reply::forgot(done.removed, true)
            }
            Err(e) => {
                tracing::warn!(error = %e, user = %cmd.user.id, "forget failed");
                self.record(
                    event("discord_error", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("stage", "forget")
                        .with("kind", error_kind(&e)),
                );
                reply::memory_failure(&e)
            }
        };
        finish(ctx, cmd, vec![text]).await;
    }

    /// Answers the forget everything prompt. Only the user who asked may press it.
    pub(super) async fn forget_pressed(
        &self,
        ctx: &Context,
        press: &ComponentInteraction,
        id: CustomId,
    ) {
        let erase = matches!(id, CustomId::ForgetAll { .. });
        if !id.may_press(press.user.id.get()) {
            let msg = CreateInteractionResponseMessage::new()
                .content("That button is not yours.")
                .ephemeral(true);
            if let Err(e) = press
                .create_response(&ctx.http, CreateInteractionResponse::Message(msg))
                .await
            {
                tracing::warn!(error = %e, "refusal response failed");
            }
            return;
        }
        if let Err(e) = press.defer(&ctx.http).await {
            tracing::warn!(error = %e, "defer failed");
            return;
        }
        let text = match (erase, press.guild_id) {
            (false, _) => "Kept everything.".to_owned(),
            (true, None) => "Use this in the server.".to_owned(),
            (true, Some(guild_id)) => {
                let req = ForgetRequest {
                    user_id: press.user.id.to_string(),
                    tenant_id: guild_id.to_string(),
                    label: None,
                };
                let span = tracing::info_span!(
                    "discord.forget",
                    "discord.command" = "forget",
                    "posthog.distinct_id" = %press.user.id,
                    "sparky.everything" = true,
                );
                match self.engine.forget(&req).instrument(span).await {
                    Ok(done) => {
                        tracing::info!(user = %press.user.id, removed = done.removed, "forgot everything");
                        self.record(
                            event(
                                "discord_forget",
                                press.user.id,
                                press.guild_id,
                                press.channel_id,
                            )
                            .with("removed", done.removed)
                            .with("everything", true),
                        );
                        reply::forgot(done.removed, false)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, user = %press.user.id, "forget failed");
                        self.record(
                            event(
                                "discord_error",
                                press.user.id,
                                press.guild_id,
                                press.channel_id,
                            )
                            .with("stage", "forget")
                            .with("kind", error_kind(&e)),
                        );
                        reply::memory_failure(&e)
                    }
                }
            }
        };
        let edit = EditInteractionResponse::new()
            .content(text)
            .components(Vec::new());
        if let Err(e) = press.edit_response(&ctx.http, edit).await {
            tracing::warn!(error = %e, "could not close the forget prompt");
        }
    }
}

/// Defers a command so only the caller sees the reply. False when Discord refused.
async fn defer_private(ctx: &Context, cmd: &CommandInteraction) -> bool {
    match cmd.defer_ephemeral(&ctx.http).await {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(error = %e, "defer failed");
            false
        }
    }
}

/// Fills a deferred private response: first message in place, rest as private followups.
async fn finish(ctx: &Context, cmd: &CommandInteraction, messages: Vec<String>) {
    let mut messages = messages.into_iter();
    let first = messages.next().unwrap_or_default();
    if let Err(e) = cmd
        .edit_response(&ctx.http, EditInteractionResponse::new().content(first))
        .await
    {
        tracing::warn!(error = %e, "response edit failed");
    }
    for more in messages {
        let followup = CreateInteractionResponseFollowup::new()
            .content(more)
            .ephemeral(true);
        if let Err(e) = cmd.create_followup(&ctx.http, followup).await {
            tracing::warn!(error = %e, "followup failed");
        }
    }
}
