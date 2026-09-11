//! One streamed turn, shown as a single message edited in place.

use std::time::Instant;

use serenity::all::{Context, MessageId};
use tokio::sync::mpsc::UnboundedReceiver;
use tracing::Instrument;

use super::Handler;
use super::destination::Destination;
use crate::core::types::{ChatRequest, ChatResponse, EngineError, Update};
use crate::render::card::{self, Pacer, Steps};
use crate::render::components::{self, ButtonSpec};
use crate::render::reply;

impl Handler {
    /// Runs req against the engine and shows the turn on one message in dest. span receives
    /// the session id and the answer.
    pub(super) async fn converse(
        &self,
        ctx: &Context,
        dest: &Destination<'_>,
        req: &ChatRequest,
        span: tracing::Span,
    ) {
        let limit = self.max_message_chars;
        let posted = dest
            .send(&ctx.http, card::thinking(&[], limit), Vec::new(), true)
            .await;
        let card_id = match posted {
            Ok(message) => Some(message.id),
            Err(e) => {
                tracing::warn!(error = %e, "could not post the turn message");
                None
            }
        };
        let mut steps = Steps::default();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let streaming = self.engine.chat_stream(req, tx).instrument(span.clone());
        let outcome = tokio::join!(
            streaming,
            self.watch(ctx, dest, card_id, &mut rx, &mut steps)
        )
        .1;
        let outcome = outcome.unwrap_or_else(|| {
            tracing::error!(user = %req.user_id, "stream ended with no outcome");
            Err(EngineError::Transport("no answer".into()))
        });
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
                    user = %req.user_id,
                    "answered"
                );
                let messages = card::answer(steps.lines(), &resp, limit);
                let rows = components::rows_for(&resp);
                self.show(ctx, dest, card_id, messages, &rows).await;
            }
            Err(e) => {
                tracing::error!(error = %e, user = %req.user_id, "engine call failed");
                let text = card::failed(steps.lines(), &reply::failure(&e), limit);
                self.show(ctx, dest, card_id, vec![text], &[]).await;
            }
        }
    }

    /// Puts the final render on the card, the first message in place and the rest as
    /// continuations. Buttons go on the first. When the card cannot be edited, the render is
    /// posted anew and the stale card is deleted.
    async fn show(
        &self,
        ctx: &Context,
        dest: &Destination<'_>,
        card_id: Option<MessageId>,
        messages: Vec<String>,
        rows: &[Vec<ButtonSpec>],
    ) {
        let mut messages = messages.into_iter();
        let Some(first) = messages.next() else {
            return;
        };
        let edited = match card_id {
            Some(id) => dest
                .edit(
                    &ctx.http,
                    id,
                    first.clone(),
                    Some(components::to_action_rows(rows)),
                )
                .await
                .map(|_| ()),
            None => Err(serenity::Error::Other("no turn message")),
        };
        if let Err(edit_error) = edited {
            let sent = dest
                .send(&ctx.http, first, components::to_action_rows(rows), true)
                .await;
            match (sent, card_id) {
                (Ok(_), Some(stale)) => {
                    if let Err(e) = dest.delete(&ctx.http, stale).await {
                        tracing::warn!(error = %e, "could not delete the stale turn message");
                    }
                }
                (Ok(_), None) => {}
                (Err(e), _) => {
                    tracing::error!(
                        edit_error = %edit_error,
                        send_error = %e,
                        "the answer could not be shown"
                    );
                }
            }
        }
        for more in messages {
            if let Err(e) = dest.send(&ctx.http, more, Vec::new(), false).await {
                tracing::warn!(error = %e, "continuation message failed");
            }
        }
    }

    /// Collects steps while the turn runs and edits the card at most once per edit_every.
    /// A step that arrives inside the gap is shown on the next edit. Returns the outcome.
    async fn watch(
        &self,
        ctx: &Context,
        dest: &Destination<'_>,
        card_id: Option<MessageId>,
        rx: &mut UnboundedReceiver<Update>,
        steps: &mut Steps,
    ) -> Option<Result<ChatResponse, EngineError>> {
        let mut outcome = None;
        let mut pacer = Pacer::new(self.edit_every, Instant::now());
        loop {
            let wait = pacer.wait(Instant::now()).filter(|_| outcome.is_none());
            tokio::select! {
                update = rx.recv() => match update {
                    None => break,
                    Some(Update::Progress(text)) => pacer.mark(steps.push(&text)),
                    Some(Update::Answer(answer)) => outcome = Some(Ok(*answer)),
                    Some(Update::Failed(e)) => outcome = Some(Err(e)),
                },
                () = tokio::time::sleep(wait.unwrap_or_default()), if wait.is_some() => {
                    if let Some(id) = card_id {
                        let text = card::thinking(steps.lines(), self.max_message_chars);
                        if let Err(e) = dest.edit(&ctx.http, id, text, None).await {
                            tracing::warn!(error = %e, "progress edit failed");
                        }
                    }
                    pacer.flushed(Instant::now());
                }
            }
        }
        outcome
    }
}
