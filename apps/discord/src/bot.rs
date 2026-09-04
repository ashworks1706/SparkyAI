//! serenity client setup and event handler.

use secrecy::ExposeSecret;
use serenity::all::{
    Client, CommandInteraction, ComponentInteraction, Context, CreateInteractionResponse,
    CreateInteractionResponseFollowup, CreateInteractionResponseMessage, EventHandler,
    GatewayIntents, GuildId, Interaction, Ready, ResolvedValue, UserId,
};
use serenity::async_trait;
use tokio::sync::Mutex;
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
    /// Role names the member holds, resolved against the guild.
    async fn role_names(&self, ctx: &Context, cmd: &CommandInteraction) -> Vec<String> {
        let Some(member) = &cmd.member else {
            return Vec::new();
        };
        let Ok(roles) = self.guild_id.roles(&ctx.http).await else {
            return Vec::new();
        };
        member
            .roles
            .iter()
            .filter_map(|id| roles.get(id).map(|r| r.name.clone()))
            .collect()
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
            self.followup(ctx, cmd, "Ask me something.".into(), Vec::new())
                .await;
            return;
        }
        let conversation_id = self.conversations.lock().await.get(&cmd.user.id).copied();
        let req = ChatRequest {
            user_id: cmd.user.id.to_string(),
            tenant_id: cmd.guild_id.unwrap_or(self.guild_id).to_string(),
            channel_id: cmd.channel_id.to_string(),
            roles: self.role_names(ctx, cmd).await,
            conversation_id,
            message: question,
        };
        match self.engine.chat(&req).await {
            Ok(resp) => {
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
                let components = resp
                    .confirmation
                    .as_ref()
                    .map(reply::confirmation_row)
                    .unwrap_or_default();
                let mut messages = reply::render(&resp).into_iter();
                if let Some(first) = messages.next() {
                    self.followup(ctx, cmd, first, components).await;
                }
                for more in messages {
                    self.followup(ctx, cmd, more, Vec::new()).await;
                }
            }
            Err(e) => {
                tracing::error!(error = %e, user = %cmd.user.id, "engine call failed");
                self.followup(
                    ctx,
                    cmd,
                    "Sparky is unavailable right now. Please try again shortly.".into(),
                    Vec::new(),
                )
                .await;
            }
        }
    }

    async fn followup(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        content: String,
        components: Vec<serenity::all::CreateActionRow>,
    ) {
        let mut builder = CreateInteractionResponseFollowup::new().content(content);
        if !components.is_empty() {
            builder = builder.components(components);
        }
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

    async fn component(&self, ctx: &Context, comp: &ComponentInteraction) {
        let (action, token) = comp
            .data
            .custom_id
            .split_once(':')
            .unwrap_or((comp.data.custom_id.as_str(), ""));
        let text = match action {
            "confirm" => format!(
                "Confirmed `{token}`. Executing confirmed actions arrives with the Phase 3 confirm endpoint."
            ),
            "cancel" => "Cancelled. Nothing was sent.".to_owned(),
            _ => "Unknown action.".to_owned(),
        };
        let msg = CreateInteractionResponseMessage::new()
            .content(text)
            .ephemeral(true);
        if let Err(e) = comp
            .create_response(&ctx.http, CreateInteractionResponse::Message(msg))
            .await
        {
            tracing::warn!(error = %e, "component response failed");
        }
    }
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
        match interaction {
            Interaction::Command(cmd) => match cmd.data.name.as_str() {
                commands::ASK => self.ask(&ctx, &cmd).await,
                commands::RESET => self.reset(&ctx, &cmd).await,
                other => tracing::warn!(command = other, "unknown command"),
            },
            Interaction::Component(comp) => self.component(&ctx, &comp).await,
            _ => {}
        }
    }
}
