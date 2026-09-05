//! serenity client setup and event handler.

use secrecy::ExposeSecret;
use std::time::{Duration, Instant};

use serenity::all::{
    Client, CommandInteraction, ComponentInteraction, Context, CreateInteractionResponse,
    CreateInteractionResponseFollowup, CreateInteractionResponseMessage, EditInteractionResponse,
    EventHandler, GatewayIntents, GuildId, Interaction, Message, Permissions, Ready, ResolvedValue,
    UserId,
};
use serenity::async_trait;
use tokio::sync::Mutex;
use tracing::Instrument;
use tracing::field::Empty;
use uuid::Uuid;

use crate::commands;
use crate::components;
use crate::core::config::Config;
use crate::core::types::{ChatRequest, ChatResponse, ConfirmRequest, EngineError, Update};
use crate::engine_client::EngineClient;
use crate::reply;

/// Shortest gap between edits of the progress message; Discord throttles faster than this.
const EDIT_EVERY: Duration = Duration::from_millis(1_500);

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
        let Some(guild_id) = cmd.guild_id else {
            self.followup(ctx, cmd, "Ask me in the server, not in a DM.".into())
                .await;
            return;
        };
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
            tenant_id: guild_id.to_string(),
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
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let streaming = self.engine.chat_stream(&req, tx).instrument(span.clone());
        let (outcome, note) = tokio::join!(streaming, self.watch(ctx, cmd, &mut rx)).1;
        if let Some(note) = note
            && let Err(e) = cmd.delete_followup(&ctx.http, note.id).await
        {
            // The progress line stays above the answer; say why rather than leaving a mystery.
            tracing::warn!(error = %e, "could not clear the progress message");
        }
        let Some(outcome) = outcome else {
            tracing::error!(user = %cmd.user.id, "stream ended with no outcome");
            self.followup(
                ctx,
                cmd,
                reply::failure(&EngineError::Transport("no answer".into())),
            )
            .await;
            return;
        };
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
                self.answer(ctx, cmd, &resp).await;
            }
            Err(e) => {
                tracing::error!(error = %e, user = %cmd.user.id, "engine call failed");
                self.followup(ctx, cmd, reply::failure(&e)).await;
            }
        }
    }

    /// Sends the answer, with the buttons it asks for on the message that asks.
    async fn answer(&self, ctx: &Context, cmd: &CommandInteraction, resp: &ChatResponse) {
        let rows = components::rows_for(resp);
        let messages = reply::render(resp);
        let last = messages.len().saturating_sub(1);
        for (i, message) in messages.into_iter().enumerate() {
            let builder = CreateInteractionResponseFollowup::new().content(message);
            let builder = if i == last && !rows.is_empty() {
                builder.components(components::to_action_rows(&rows))
            } else {
                builder
            };
            if let Err(e) = cmd.create_followup(&ctx.http, builder).await {
                tracing::warn!(error = %e, "followup failed");
            }
        }
    }

    /// Relays progress while the turn runs, and returns the outcome plus the progress message
    /// still on screen, if any.
    async fn watch(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<Update>,
    ) -> (Option<Result<ChatResponse, EngineError>>, Option<Message>) {
        let mut outcome = None;
        let mut note: Option<Message> = None;
        let mut last_edit: Option<Instant> = None;
        while let Some(update) = rx.recv().await {
            match update {
                Update::Progress(text) => {
                    // Discord throttles edits, so a burst of steps collapses into one.
                    if last_edit.is_some_and(|at| at.elapsed() < EDIT_EVERY) {
                        continue;
                    }
                    last_edit = Some(Instant::now());
                    note = self.progress(ctx, cmd, note, &text).await;
                }
                Update::Answer(answer) => outcome = Some(Ok(*answer)),
                Update::Failed(e) => outcome = Some(Err(e)),
            }
        }
        (outcome, note)
    }

    /// Shows what the agent is doing, as one message edited in place. Returns it so the next
    /// step edits the same message and the answer can clear it.
    async fn progress(
        &self,
        ctx: &Context,
        cmd: &CommandInteraction,
        existing: Option<Message>,
        text: &str,
    ) -> Option<Message> {
        let body = format!("_{text}…_");
        let builder = CreateInteractionResponseFollowup::new().content(body);
        if let Some(message) = existing {
            match cmd.edit_followup(&ctx.http, message.id, builder).await {
                Ok(updated) => Some(updated),
                Err(e) => {
                    tracing::warn!(error = %e, "progress edit failed");
                    // Keep the message we have; a fresh one would leave two on screen.
                    Some(message)
                }
            }
        } else {
            match cmd.create_followup(&ctx.http, builder).await {
                Ok(posted) => Some(posted),
                Err(e) => {
                    tracing::warn!(error = %e, "progress message failed");
                    None
                }
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

impl Handler {
    /// Answers a pressed button. The engine decides whether this caller may: it only accepts
    /// the one it asked, so a bystander pressing Approve changes nothing.
    async fn pressed(&self, ctx: &Context, press: &ComponentInteraction) {
        let Some(id) = components::CustomId::parse(&press.data.custom_id) else {
            tracing::debug!(custom_id = %press.data.custom_id, "component is not ours");
            return;
        };
        if let Err(e) = press.defer(&ctx.http).await {
            tracing::warn!(error = %e, "defer failed");
            return;
        }
        let Some(guild_id) = press.guild_id else {
            return;
        };
        let approve = id.action == components::Action::Approve;
        let req = ConfirmRequest {
            token: id.token,
            approve,
            user_id: press.user.id.to_string(),
            tenant_id: guild_id.to_string(),
            conversation_id: id.conversation,
        };
        let span = tracing::info_span!(
            "discord.confirm",
            "openinference.span.kind" = "CHAIN",
            "user.id" = %press.user.id,
            "sparky.approved" = approve,
        );
        let outcome = self.engine.confirm(&req).instrument(span).await;
        let body = match outcome {
            Ok(resp) => reply::render(&resp).join("\n"),
            Err(e) => {
                tracing::error!(error = %e, user = %press.user.id, "confirm failed");
                reply::failure(&e)
            }
        };
        // The buttons are spent either way; clearing them stops a second press.
        let edit = EditInteractionResponse::new().components(Vec::new());
        if let Err(e) = press.edit_response(&ctx.http, edit).await {
            tracing::warn!(error = %e, "could not clear the buttons");
        }
        let followup = CreateInteractionResponseFollowup::new().content(body);
        if let Err(e) = press.create_followup(&ctx.http, followup).await {
            tracing::warn!(error = %e, "confirm followup failed");
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
        if let Interaction::Component(press) = &interaction {
            self.pressed(&ctx, press).await;
            return;
        }
        if let Interaction::Command(cmd) = interaction {
            match cmd.data.name.as_str() {
                commands::ASK => self.ask(&ctx, &cmd).await,
                commands::RESET => self.reset(&ctx, &cmd).await,
                other => tracing::warn!(command = other, "unknown command"),
            }
        }
    }
}
