//! Where the messages of one turn go: interaction followups, or a channel.

use serenity::all::{
    ChannelId, CommandInteraction, CreateActionRow, CreateAllowedMentions,
    CreateInteractionResponseFollowup, CreateMessage, EditMessage, Http, Message, MessageId,
};

/// The target of every message a turn posts.
pub(super) enum Destination<'a> {
    /// Followups on the command, visible only to the asker when ephemeral.
    Followup {
        /// The command being answered.
        cmd: &'a CommandInteraction,
        /// Whether only the asker sees them.
        ephemeral: bool,
    },
    /// Plain messages in a channel or thread.
    Channel {
        /// Where to post.
        channel: ChannelId,
        /// The message the first reply points at, if any.
        reply_to: Option<MessageId>,
    },
}

impl Destination<'_> {
    /// Posts one message with rows of buttons. reply marks it as a reply to reply_to.
    pub(super) async fn send(
        &self,
        http: &Http,
        content: String,
        rows: Vec<CreateActionRow>,
        reply: bool,
    ) -> serenity::Result<Message> {
        match self {
            Self::Followup { cmd, ephemeral } => {
                let mut builder = CreateInteractionResponseFollowup::new()
                    .content(content)
                    .ephemeral(*ephemeral)
                    .allowed_mentions(CreateAllowedMentions::new());
                if !rows.is_empty() {
                    builder = builder.components(rows);
                }
                cmd.create_followup(http, builder).await
            }
            Self::Channel { channel, reply_to } => {
                let mut builder = CreateMessage::new()
                    .content(content)
                    .allowed_mentions(CreateAllowedMentions::new().replied_user(true));
                if !rows.is_empty() {
                    builder = builder.components(rows);
                }
                if reply && let Some(to) = reply_to {
                    builder = builder.reference_message((*channel, *to));
                }
                channel.send_message(http, builder).await
            }
        }
    }

    /// Replaces the text of a message this destination posted. Some rows replace its buttons,
    /// an empty list clears them, and None leaves them.
    pub(super) async fn edit(
        &self,
        http: &Http,
        id: MessageId,
        content: String,
        rows: Option<Vec<CreateActionRow>>,
    ) -> serenity::Result<Message> {
        match self {
            Self::Followup { cmd, .. } => {
                let mut builder = CreateInteractionResponseFollowup::new().content(content);
                if let Some(rows) = rows {
                    builder = builder.components(rows);
                }
                cmd.edit_followup(http, id, builder).await
            }
            Self::Channel { channel, .. } => {
                let mut builder = EditMessage::new().content(content);
                if let Some(rows) = rows {
                    builder = builder.components(rows);
                }
                channel.edit_message(http, id, builder).await
            }
        }
    }

    /// Deletes a message this destination posted.
    pub(super) async fn delete(&self, http: &Http, id: MessageId) -> serenity::Result<()> {
        match self {
            Self::Followup { cmd, .. } => cmd.delete_followup(http, id).await,
            Self::Channel { channel, .. } => channel.delete_message(http, id).await,
        }
    }

    /// Posts one plain line, logging a failure.
    pub(super) async fn say(&self, http: &Http, content: impl Into<String>) {
        if let Err(e) = self.send(http, content.into(), Vec::new(), true).await {
            tracing::warn!(error = %e, "reply failed");
        }
    }
}
