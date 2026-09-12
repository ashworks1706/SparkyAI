//! Questions asked by mentioning the bot in a message.

use std::collections::HashMap;

use serenity::all::{Context, Message, Permissions, RoleId, UserId};
use tracing::field::Empty;

use super::destination::Destination;
use super::{Handler, event};
use crate::access::roles::{authorized_roles, member_permissions};
use crate::access::route::{self, Place};
use crate::render::reply;

impl Handler {
    /// Answers a guild message that mentions the bot. Everything else is ignored.
    pub(super) async fn mentioned(&self, ctx: &Context, msg: &Message) {
        if msg.author.bot {
            return;
        }
        let Some(guild_id) = msg.guild_id else {
            return;
        };
        let Some(&me) = self.me.get() else {
            return;
        };
        if guild_id != self.guild_id || !msg.mentions_user_id(me) {
            return;
        }
        let here = Destination::Channel {
            channel: msg.channel_id,
            reply_to: Some(msg.id),
        };
        let channel = match msg.channel_id.to_channel(ctx).await {
            Ok(channel) => channel.guild(),
            Err(e) => {
                tracing::warn!(error = %e, channel = %msg.channel_id, "channel lookup failed");
                here.say(&ctx.http, reply::UNAVAILABLE).await;
                return;
            }
        };
        let Some(channel) = channel else {
            return;
        };
        let in_thread = route::is_thread(channel.kind);
        let parent = route::thread_parent(Some(channel.kind), channel.parent_id);
        if !self.serves(msg.channel_id, parent) {
            return;
        }
        if !self.within_cooldown(msg.author.id).await {
            here.say(&ctx.http, self.cooldown_text()).await;
            return;
        }
        let question = route::strip_mentions(&msg.content, me);
        if question.is_empty() {
            here.say(&ctx.http, "Ask me something.").await;
            return;
        }
        let label = route::place_name(false, in_thread);
        self.record(
            event(
                "discord_mention",
                msg.author.id,
                Some(guild_id),
                msg.channel_id,
            )
            .with("place", label),
        );
        let roles = match self.message_roles(ctx, msg).await {
            Ok(roles) => roles,
            Err(e) => {
                tracing::error!(error = %e, user = %msg.author.id, "role lookup failed");
                self.record(
                    event(
                        "discord_error",
                        msg.author.id,
                        Some(guild_id),
                        msg.channel_id,
                    )
                    .with("stage", "roles")
                    .with("kind", "discord")
                    .with("place", label),
                );
                here.say(
                    &ctx.http,
                    "I could not verify your roles, so I did not run that.",
                )
                .await;
                return;
            }
        };
        let (dest, place) = if in_thread {
            (here, Place::Thread(msg.channel_id))
        } else {
            self.mention_thread(ctx, msg, &question, here).await
        };
        let req = route::chat_request(place, msg.author.id, guild_id, roles, question);
        let span = tracing::info_span!(
            "discord.mention",
            "discord.command" = "mention",
            "posthog.distinct_id" = %msg.author.id,
            "sparky.visibility" = ?req.visibility,
            "$ai_session_id" = Empty,
            "sparky.input" = %req.message,
            "sparky.output" = Empty,
            // OpenInference, read by the Phoenix trace UI.
            "openinference.span.kind" = "CHAIN",
            "input.value" = %req.message,
            "output.value" = Empty,
            "session.id" = Empty,
            "user.id" = %msg.author.id,
        );
        self.converse(ctx, &dest, &req, span, label).await;
    }

    /// Opens a thread from msg for question. Falls back to here when the thread cannot be made.
    async fn mention_thread<'a>(
        &self,
        ctx: &Context,
        msg: &Message,
        question: &str,
        here: Destination<'a>,
    ) -> (Destination<'a>, Place) {
        match msg
            .channel_id
            .create_thread_from_message(&ctx.http, msg.id, self.thread_for(question))
            .await
        {
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
                    channel = %msg.channel_id,
                    "could not open a thread; answering inline"
                );
                (here, Place::Inline(msg.channel_id))
            }
        }
    }

    /// Role names the author holds plus the capability their guild roles or ownership grant.
    /// A lookup failure is an error, not an empty list.
    async fn message_roles(
        &self,
        ctx: &Context,
        msg: &Message,
    ) -> Result<Vec<String>, serenity::Error> {
        let held: Vec<RoleId> = msg
            .member
            .as_ref()
            .map(|m| m.roles.clone())
            .unwrap_or_default();
        let owner = self.owner(ctx).await?;
        let roles = self.guild_id.roles(&ctx.http).await?;
        let bits: HashMap<RoleId, Permissions> =
            roles.iter().map(|(id, r)| (*id, r.permissions)).collect();
        let everyone = RoleId::new(self.guild_id.get());
        let permissions = member_permissions(everyone, &held, &bits, msg.author.id == owner);
        let names = held
            .iter()
            .filter_map(|id| roles.get(id).map(|r| r.name.clone()));
        Ok(authorized_roles(
            names,
            Some(permissions),
            &self.write_capability,
        ))
    }

    /// The guild owner, from the cache when it holds the guild.
    async fn owner(&self, ctx: &Context) -> Result<UserId, serenity::Error> {
        let cached = ctx.cache.guild(self.guild_id).map(|g| g.owner_id);
        if let Some(owner) = cached {
            return Ok(owner);
        }
        Ok(self.guild_id.to_partial_guild(&ctx.http).await?.owner_id)
    }
}
