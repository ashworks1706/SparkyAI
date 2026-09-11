//! The approval buttons on a turn message.

use serenity::all::{
    ComponentInteraction, Context, CreateInteractionResponseFollowup, EditInteractionResponse,
};
use tracing::Instrument;
use tracing::field::Empty;
use uuid::Uuid;

use super::{Handler, error_kind, event};
use crate::access::route;
use crate::core::types::ConfirmRequest;
use crate::render::card;
use crate::render::components::{self, Action};
use crate::render::reply;

impl Handler {
    /// Answers a pressed approval button. The card changes only once the engine accepts the
    /// presser; anyone else gets a private line and the card keeps its buttons.
    pub(super) async fn confirm(
        &self,
        ctx: &Context,
        press: &ComponentInteraction,
        action: Action,
        token: Uuid,
        conversation: Uuid,
    ) {
        if let Err(e) = press.defer(&ctx.http).await {
            tracing::warn!(error = %e, "defer failed");
            return;
        }
        let Some(guild_id) = press.guild_id else {
            return;
        };
        let approve = action == Action::Approve;
        let (visibility, ephemeral) = route::press_visibility(press.message.flags);
        let req = ConfirmRequest {
            token,
            approve,
            user_id: press.user.id.to_string(),
            tenant_id: guild_id.to_string(),
            conversation_id: conversation,
            visibility,
        };
        let span = tracing::info_span!(
            "discord.confirm",
            "discord.command" = "confirm",
            "posthog.distinct_id" = %press.user.id,
            "$ai_session_id" = %conversation,
            "sparky.approved" = approve,
            "sparky.visibility" = ?visibility,
            "sparky.output" = Empty,
        );
        let answered = self.engine.confirm(&req).instrument(span.clone()).await;
        let resp = match answered {
            Ok(resp) => {
                span.record(
                    "sparky.output",
                    resp.text.chars().take(2_000).collect::<String>().as_str(),
                );
                self.record(
                    event(
                        "discord_confirm",
                        press.user.id,
                        Some(guild_id),
                        press.channel_id,
                    )
                    .with("$session_id", conversation.to_string())
                    .with("conversation_id", conversation.to_string())
                    .with("approved", approve)
                    .with("status", resp.status.as_str()),
                );
                resp
            }
            Err(e) => {
                tracing::warn!(error = %e, user = %press.user.id, "confirm refused or failed");
                self.record(
                    event(
                        "discord_error",
                        press.user.id,
                        Some(guild_id),
                        press.channel_id,
                    )
                    .with("stage", "confirm")
                    .with("kind", error_kind(&e)),
                );
                self.tell_presser(ctx, press, reply::confirm_failure(&e))
                    .await;
                return;
            }
        };
        let messages = card::resumed(
            &press.message.content,
            approve,
            &resp,
            self.max_message_chars,
        );
        let rows = components::rows_for(&resp);
        let mut messages = messages.into_iter();
        let first = messages.next().unwrap_or_default();
        let edit = EditInteractionResponse::new()
            .content(first)
            .components(components::to_action_rows(&rows));
        if let Err(e) = press.edit_response(&ctx.http, edit).await {
            tracing::warn!(error = %e, "could not show the resumed answer");
        }
        for more in messages {
            let followup = CreateInteractionResponseFollowup::new()
                .content(more)
                .ephemeral(ephemeral);
            if let Err(e) = press.create_followup(&ctx.http, followup).await {
                tracing::warn!(error = %e, "continuation followup failed");
            }
        }
    }

    /// Sends the presser a line only they see.
    async fn tell_presser(&self, ctx: &Context, press: &ComponentInteraction, text: String) {
        let followup = CreateInteractionResponseFollowup::new()
            .content(text)
            .ephemeral(true);
        if let Err(e) = press.create_followup(&ctx.http, followup).await {
            tracing::warn!(error = %e, "presser followup failed");
        }
    }
}
