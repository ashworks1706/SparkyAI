//! serenity client setup and event handler.

use secrecy::ExposeSecret;
use serenity::all::{
    Client, CommandInteraction, Context, CreateInteractionResponse,
    CreateInteractionResponseFollowup, CreateInteractionResponseMessage, EventHandler,
    GatewayIntents, GuildId, Interaction, Permissions, Ready, ResolvedValue, UserId,
};
use serenity::async_trait;
use tokio::sync::Mutex;
use tracing::Instrument;
use tracing::field::Empty;
use uuid::Uuid;

use crate::commands;
use crate::core::config::Config;
use crate::core::types::ChatRequest;
use crate::engine_client::EngineClient;
use crate::reply;

/// Per-process bot state and the serenity event handler.
struct Handler {
    engine: EngineClient,
    guild_id: GuildId,
    /// Conversation each user is continuing. Lost on restart; `/reset` clears it.
    conversations: Mutex<std::collections::HashMap<UserId, Uuid>>,
}

/// Connects to Discord and runs until shutdown.
pub async fn run(cfg: Config) -> anyhow::Result<()> {
    if cfg.discord.guild_id == 0 {
        anyhow::bail!("SPARKY_DISCORD__GUILD_ID is unset; set it to the guild the bot serves");
    }
    if cfg.discord.token.expose_secret().trim().is_empty() {
        anyhow::bail!("SPARKY_DISCORD__TOKEN is unset; create a bot at discord.com/developers");
    }
    let engine = EngineClient::new(&cfg.engine.base_url, cfg.engine.service_token.clone())?;
    let handler = Handler {
        engine,
        guild_id: GuildId::new(cfg.discord.guild_id),
        conversations: Mutex::new(std::collections::HashMap::new()),
    };
    let intents = GatewayIntents::non_privileged();
    let mut client = Client::builder(cfg.discord.token.expose_secret(), intents)
        .event_handler(handler)
        .await?;
    client.start().await?;
    Ok(())
}

impl Handler {
    /// Role names the member holds, resolved against the guild. A lookup failure is an
    /// error, not an empty list: permissions must never silently drop.
    async fn role_names(
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
        Ok(authorized_roles(names, member.permissions))
    }

    async fn ask(&self, ctx: &Context, cmd: &CommandInteraction) {
        if let Err(e) = cmd.defer(&ctx.http).await {
            tracing::warn!(error = %e, "defer failed");
            return;
        }
        let question = cmd
            .data
            .options()
            .into_iter()
            .find(|o| o.name == commands::QUESTION)
            .and_then(|o| match o.value {
                ResolvedValue::String(s) => Some(s.to_owned()),
                _ => None,
            })
            .unwrap_or_default();
        if question.trim().is_empty() {
            self.followup(ctx, cmd, "Ask me something.".into()).await;
            return;
        }
        let roles = match self.role_names(ctx, cmd).await {
            Ok(roles) => roles,
            Err(e) => {
                tracing::error!(error = %e, user = %cmd.user.id, "role lookup failed");
                self.followup(
                    ctx,
                    cmd,
                    "I could not verify your roles, so I did not run that.".into(),
                )
                .await;
                return;
            }
        };
        let conversation_id = self.conversations.lock().await.get(&cmd.user.id).copied();
        let req = ChatRequest {
            user_id: cmd.user.id.to_string(),
            tenant_id: cmd.guild_id.unwrap_or(self.guild_id).to_string(),
            channel_id: cmd.channel_id.to_string(),
            roles,
            conversation_id,
            message: question,
        };
        let span = tracing::info_span!(
            "discord.ask",
            "openinference.span.kind" = "CHAIN",
            "user.id" = %cmd.user.id,
            "session.id" = Empty,
            "input.value" = %req.message,
            "output.value" = Empty,
        );
        let outcome = self.engine.chat(&req).instrument(span.clone()).await;
        match outcome {
            Ok(resp) => {
                span.record("session.id", resp.conversation_id.to_string().as_str());
                span.record(
                    "output.value",
                    resp.text.chars().take(2_000).collect::<String>().as_str(),
                );
                tracing::info!(
                    request_id = %resp.request_id,
                    status = %resp.status,
                    user = %cmd.user.id,
                    "answered"
                );
                self.conversations
                    .lock()
                    .await
                    .insert(cmd.user.id, resp.conversation_id);
                for message in reply::render(&resp) {
                    self.followup(ctx, cmd, message).await;
                }
            }
            Err(e) => {
                tracing::error!(error = %e, user = %cmd.user.id, "engine call failed");
                self.followup(ctx, cmd, reply::failure(&e)).await;
            }
        }
    }

    async fn followup(&self, ctx: &Context, cmd: &CommandInteraction, content: String) {
        let builder = CreateInteractionResponseFollowup::new().content(content);
        if let Err(e) = cmd.create_followup(&ctx.http, builder).await {
            tracing::warn!(error = %e, "followup failed");
        }
    }

    async fn reset(&self, ctx: &Context, cmd: &CommandInteraction) {
        self.conversations.lock().await.remove(&cmd.user.id);
        let msg = CreateInteractionResponseMessage::new()
            .content("Fresh start. Ask away.")
            .ephemeral(true);
        if let Err(e) = cmd
            .create_response(&ctx.http, CreateInteractionResponse::Message(msg))
            .await
        {
            tracing::warn!(error = %e, "reset response failed");
        }
    }
}

/// The marker the engine's policy reads to allow write-side tools.
pub(crate) const WRITE_CAPABILITY: &str = "MANAGE_GUILD";

/// Whether a member's own Discord permissions let them ask for write-side tools.
pub(crate) fn can_write(permissions: Permissions) -> bool {
    permissions.intersects(Permissions::MANAGE_GUILD | Permissions::ADMINISTRATOR)
}

/// Guild role names plus the write marker when the member's own Discord permissions grant it.
/// A guild role named like the marker is dropped: only the permission bits confer write access.
pub(crate) fn authorized_roles(
    names: impl IntoIterator<Item = String>,
    permissions: Option<Permissions>,
) -> Vec<String> {
    let mut roles: Vec<String> = names
        .into_iter()
        .filter(|n| n != WRITE_CAPABILITY)
        .collect();
    if permissions.is_some_and(can_write) {
        roles.push(WRITE_CAPABILITY.to_owned());
    }
    roles
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        tracing::info!(user = %ready.user.name, guild = %self.guild_id, "connected");
        match self.guild_id.set_commands(&ctx.http, commands::all()).await {
            Ok(cmds) => tracing::info!(count = cmds.len(), "commands registered"),
            Err(e) => tracing::error!(error = %e, "command registration failed"),
        }
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        if let Interaction::Command(cmd) = interaction {
            match cmd.data.name.as_str() {
                commands::ASK => self.ask(&ctx, &cmd).await,
                commands::RESET => self.reset(&ctx, &cmd).await,
                other => tracing::warn!(command = other, "unknown command"),
            }
        }
    }
}
