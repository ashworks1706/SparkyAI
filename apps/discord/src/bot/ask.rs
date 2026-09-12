//! The /ask command: private followups, followups in a thread, or a new thread.

use serenity::all::{CommandInteraction, Context, CreateAllowedMentions, EditInteractionResponse};
use tracing::field::Empty;

use super::commands;
use super::destination::Destination;
use super::{Handler, event, option_bool, option_str, tell};
use crate::access::roles::authorized_roles;
use crate::access::route::{self, AskMode, Place};

impl Handler {
    /// Answers one /ask.
    pub(super) async fn ask(&self, ctx: &Context, cmd: &CommandInteraction) {
        let kind = cmd.channel.as_ref().map(|c| c.kind);
        let parent = route::thread_parent(kind, cmd.channel.as_ref().and_then(|c| c.parent_id));
        if !self.serves(cmd.channel_id, parent) {
            tell(ctx, cmd, "I do not answer in this channel.").await;
            return;
        }
        if !self.within_cooldown(cmd.user.id).await {
            tell(ctx, cmd, self.cooldown_text()).await;
            return;
        }
        let mode = route::ask_mode(option_bool(cmd, commands::PRIVATE), kind);
        let deferred = if mode == AskMode::Private {
            cmd.defer_ephemeral(&ctx.http).await
        } else {
            cmd.defer(&ctx.http).await
        };
        if let Err(e) = deferred {
            tracing::warn!(error = %e, "defer failed");
            return;
        }
        let inline = Destination::Followup {
            cmd,
            ephemeral: mode == AskMode::Private,
        };
        let question = option_str(cmd, commands::QUESTION).unwrap_or_default();
        if question.trim().is_empty() {
            inline.say(&ctx.http, "Ask me something.").await;
            return;
        }
        let label = route::place_name(cmd.guild_id.is_none(), kind.is_some_and(route::is_thread));
        self.record(
            event("discord_ask", cmd.user.id, cmd.guild_id, cmd.channel_id)
                .with("place", label)
                .with("private", mode == AskMode::Private),
        );
        let Some(guild_id) = cmd.guild_id else {
            inline
                .say(&ctx.http, "Ask me in the server, not in a DM.")
                .await;
            return;
        };
        let roles = match self.command_roles(ctx, cmd).await {
            Ok(roles) => roles,
            Err(e) => {
                tracing::error!(error = %e, user = %cmd.user.id, "role lookup failed");
                self.record(
                    event("discord_error", cmd.user.id, cmd.guild_id, cmd.channel_id)
                        .with("stage", "roles")
                        .with("kind", "discord")
                        .with("place", label),
                );
                inline
                    .say(
                        &ctx.http,
                        "I could not verify your roles, so I did not run that.",
                    )
                    .await;
                return;
            }
        };
        let (dest, place) = match mode {
            AskMode::Private => (inline, Place::Private(cmd.channel_id)),
            AskMode::InThread => (inline, Place::Thread(cmd.channel_id)),
            AskMode::NewThread => self.open_thread(ctx, cmd, &question).await,
        };
        let req = route::chat_request(place, cmd.user.id, guild_id, roles, question);
        let span = tracing::info_span!(
            "discord.ask",
            "discord.command" = "ask",
            "posthog.distinct_id" = %cmd.user.id,
            "sparky.visibility" = ?req.visibility,
            "$ai_session_id" = Empty,
            "sparky.input" = %req.message,
            "sparky.output" = Empty,
            // OpenInference, read by the Phoenix trace UI.
            "openinference.span.kind" = "CHAIN",
            "input.value" = %req.message,
            "output.value" = Empty,
            "session.id" = Empty,
            "otel.status_code" = Empty,
            "otel.status_message" = Empty,
            "user.id" = %cmd.user.id,
        );
        self.converse(ctx, &dest, &req, span, label).await;
    }

    /// Shows the question as the command response and opens a thread from it. Falls back to
    /// inline followups when the thread cannot be made.
    async fn open_thread<'a>(
        &self,
        ctx: &Context,
        cmd: &'a CommandInteraction,
        question: &str,
    ) -> (Destination<'a>, Place) {
        let header = route::asked_header(cmd.user.display_name(), question, self.max_message_chars);
        let edit = EditInteractionResponse::new()
            .content(header)
            .allowed_mentions(CreateAllowedMentions::new());
        let thread = match cmd.edit_response(&ctx.http, edit).await {
            Ok(shown) => {
                cmd.channel_id
                    .create_thread_from_message(&ctx.http, shown.id, self.thread_for(question))
                    .await
            }
            Err(e) => Err(e),
        };
        match thread {
            Ok(thread) => (
                Destination::Channel {
                    channel: thread.id,
                    reply_to: None,
                },
                Place::Thread(thread.id),
            ),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    channel = %cmd.channel_id,
                    "could not open a thread; answering inline"
                );
                (
                    Destination::Followup {
                        cmd,
                        ephemeral: false,
                    },
                    Place::Inline(cmd.channel_id),
                )
            }
        }
    }

    /// Role names the member holds, resolved against the guild, with the capability the
    /// interaction permissions grant. A lookup failure is an error, not an empty list.
    async fn command_roles(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
    ) -> Result<Vec<String>, serenity::Error> {
        let Some(member) = &cmd.member else {
            return Ok(Vec::new());
        };
        let roles = self.guild_id.roles(&ctx.http).await?;
        let names = member
            .roles
            .iter()
            .filter_map(|id| roles.get(id).map(|r| r.name.clone()));
        Ok(authorized_roles(
            names,
            member.permissions,
            &self.write_capability,
        ))
    }
}
