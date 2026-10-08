//! Private replies to a slash command: an immediate line, a deferral, and its fill.

use serenity::all::{
    CommandInteraction, Context, CreateInteractionResponse, CreateInteractionResponseFollowup,
    CreateInteractionResponseMessage, EditInteractionResponse,
};

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
