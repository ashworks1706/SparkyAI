//! Replies to an interaction: an immediate private line, a deferral, and its fill.

use serenity::all::{
    CommandInteraction, Context, CreateActionRow, CreateInteractionResponse,
    CreateInteractionResponseFollowup, CreateInteractionResponseMessage, EditInteractionResponse,
};
use serenity::builder::Builder;

/// Answers a command at once with a line only the caller sees.
pub(super) async fn tell(ctx: &Context, cmd: &CommandInteraction, text: impl Into<String>) {
    let msg = CreateInteractionResponseMessage::new()
        .content(text)
        .ephemeral(true);
    if let Err(e) = cmd
        .create_response(&ctx.http, CreateInteractionResponse::Message(msg))
        .await
    {
        tracing::warn!(error = %e, "ephemeral response failed");
    }
}

/// Defers a command so only the caller sees the reply. False when Discord refused.
pub(super) async fn defer_private(ctx: &Context, cmd: &CommandInteraction) -> bool {
    match cmd.defer_ephemeral(&ctx.http).await {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(error = %e, "defer failed");
            false
        }
    }
}

/// Fills a deferred private response: first message in place, rest as private followups.
pub(super) async fn finish(ctx: &Context, cmd: &CommandInteraction, messages: Vec<String>) {
    fill(ctx, &cmd.token, messages, None, true).await;
}

/// Fills the deferred response of the interaction with token: the first message edits it,
/// with components when given, and the rest follow up.
pub(super) async fn fill(
    ctx: &Context,
    token: &str,
    messages: Vec<String>,
    components: Option<Vec<CreateActionRow>>,
    ephemeral: bool,
) {
    let mut messages = messages.into_iter();
    let mut edit = EditInteractionResponse::new().content(messages.next().unwrap_or_default());
    if let Some(rows) = components {
        edit = edit.components(rows);
    }
    if let Err(e) = edit.execute(&ctx.http, token).await {
        tracing::warn!(error = %e, "response edit failed");
    }
    for more in messages {
        let followup = CreateInteractionResponseFollowup::new()
            .content(more)
            .ephemeral(ephemeral);
        if let Err(e) = followup.execute(&ctx.http, (None, token)).await {
            tracing::warn!(error = %e, "followup failed");
        }
    }
}
