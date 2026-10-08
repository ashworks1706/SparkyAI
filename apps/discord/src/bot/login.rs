//! The /login and /logout commands, which connect and disconnect a user account.

use serenity::all::{CommandInteraction, Context};
use tracing::Instrument;

use super::commands;
use super::respond::{defer_private, finish};
use super::{Handler, error_kind, event, option_str};
use crate::core::types::AuthorizeRequest;

impl Handler {
    /// Gives the caller a private link to connect one of the SERVICES accounts.
    pub(super) async fn login(&self, ctx: &Context, cmd: &CommandInteraction) {
        if !defer_private(ctx, cmd).await {
            return;
        }
        let Some((service, provider)) = chosen_service(cmd) else {
            finish(ctx, cmd, vec!["Choose a service to connect.".to_owned()]).await;
            return;
        };
        let req = AuthorizeRequest {
            user: cmd.user.id.to_string(),
        };
        let span = tracing::info_span!(
            "discord.login",
            "discord.command" = "login",
            "sparky.provider" = provider,
            "user.id" = %cmd.user.id,
        );
        let text = match self
            .engine
            .oauth_login(provider, &req)
            .instrument(span)
            .await
        {
            Ok(resp) => {
                tracing::info!(user = %cmd.user.id, provider, "login link issued");
                self.record(
                    event("discord_login", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("provider", provider),
                );
                format!(
                    "Connect your {service} with this link (only you can see it):\n{}\n\nYou sign \
                     in through the provider. Once connected, send me a direct message and ask \
                     about it.",
                    resp.url
                )
            }
            Err(e) => {
                tracing::warn!(error = %e, user = %cmd.user.id, provider, "login failed");
                self.record(
                    event("discord_error", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("stage", "login")
                        .with("kind", error_kind(&e)),
                );
                format!("{service} login is not available right now.")
            }
        };
        finish(ctx, cmd, vec![text]).await;
    }

    /// Disconnects one of the SERVICES accounts of the caller.
    pub(super) async fn logout(&self, ctx: &Context, cmd: &CommandInteraction) {
        if !defer_private(ctx, cmd).await {
            return;
        }
        let Some((service, provider)) = chosen_service(cmd) else {
            finish(ctx, cmd, vec!["Choose a service to disconnect.".to_owned()]).await;
            return;
        };
        let req = AuthorizeRequest {
            user: cmd.user.id.to_string(),
        };
        let span = tracing::info_span!(
            "discord.logout",
            "discord.command" = "logout",
            "sparky.provider" = provider,
            "user.id" = %cmd.user.id,
        );
        let text = match self
            .engine
            .oauth_logout(provider, &req)
            .instrument(span)
            .await
        {
            Ok(done) => {
                tracing::info!(user = %cmd.user.id, provider, removed = done.removed, "disconnected");
                self.record(
                    event("discord_logout", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("provider", provider)
                        .with("removed", done.removed),
                );
                if done.removed {
                    format!("Your {service} is disconnected.")
                } else {
                    format!("You had no {service} connected.")
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, user = %cmd.user.id, provider, "logout failed");
                self.record(
                    event("discord_error", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("stage", "logout")
                        .with("kind", error_kind(&e)),
                );
                format!("Could not disconnect {service} right now.")
            }
        };
        finish(ctx, cmd, vec![text]).await;
    }
}

/// The service the caller chose, as (display label, engine provider key).
fn chosen_service(cmd: &CommandInteraction) -> Option<(&'static str, &'static str)> {
    let value = option_str(cmd, commands::SERVICE)?;
    commands::SERVICES
        .iter()
        .find(|(_, provider)| *provider == value.as_str())
        .map(|(label, provider)| (*label, *provider))
}
