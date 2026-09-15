//! Where the messages of one turn go: a channel or the thread opened for the answer.

use serenity::all::{
    ChannelId, CreateActionRow, CreateAllowedMentions, CreateMessage, EditMessage, Http, Message,
    MessageId,
};

/// The target of every message a turn posts.
pub(super) struct Destination {
    /// Where to post.
    pub(super) channel: ChannelId,
    /// The message the first reply points at, if any.
    pub(super) reply_to: Option<MessageId>,
}

impl Destination {
    /// Posts one message with rows of buttons. reply marks it as a reply to reply_to.
    pub(super) async fn send(
        &self,
        http: &Http,
        content: String,
        rows: Vec<CreateActionRow>,
        reply: bool,
    ) -> serenity::Result<Message> {
        let mut builder = CreateMessage::new()
            .content(content)
            .allowed_mentions(CreateAllowedMentions::new().replied_user(true));
        if !rows.is_empty() {
            builder = builder.components(rows);
        }
        if reply && let Some(to) = self.reply_to {
            builder = builder.reference_message((self.channel, to));
        }
        self.channel.send_message(http, builder).await
    }

    /// Replaces a posted message's text; some rows replace buttons, empty clears, None keeps them.
    pub(super) async fn edit(
        &self,
        http: &Http,
        id: MessageId,
        content: String,
        rows: Option<Vec<CreateActionRow>>,
    ) -> serenity::Result<Message> {
        let mut builder = EditMessage::new().content(content);
        if let Some(rows) = rows {
            builder = builder.components(rows);
        }
        self.channel.edit_message(http, id, builder).await
    }

    /// Deletes a message this destination posted.
    pub(super) async fn delete(&self, http: &Http, id: MessageId) -> serenity::Result<()> {
        self.channel.delete_message(http, id).await
    }

    /// Posts one plain line, logging a failure.
    pub(super) async fn say(&self, http: &Http, content: impl Into<String>) {
        if let Err(e) = self.send(http, content.into(), Vec::new(), true).await {
            tracing::warn!(error = %e, "reply failed");
        }
    }
}
