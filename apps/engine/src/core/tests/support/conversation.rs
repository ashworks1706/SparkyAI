//! Conversation store doubles: a recorder and a store that keeps ownership the way the database does.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::core::traits::conversation::ConversationStore;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::Visibility;
use crate::core::types::conversation::message::Message;
use crate::core::types::store::StoreError;

/// A conversation store that remembers what the loop asked it to keep.
#[derive(Default)]
pub struct Recording {
    turns: Mutex<Vec<Message>>,
}

impl Recording {
    pub fn appended(&self) -> Vec<Message> {
        self.turns.lock().map(|t| t.clone()).unwrap_or_default()
    }
}

#[async_trait]
impl ConversationStore for Recording {
    async fn ensure(&self, _ctx: &RequestContext, _channel_id: &str) -> Result<(), StoreError> {
        Ok(())
    }

    async fn owns(&self, _ctx: &RequestContext) -> Result<bool, StoreError> {
        Ok(true)
    }

    async fn load(&self, _ctx: &RequestContext, _limit: usize) -> Result<Vec<Message>, StoreError> {
        Ok(Vec::new())
    }

    async fn latest(
        &self,
        _ctx: &RequestContext,
        _channel_id: &str,
    ) -> Result<Option<uuid::Uuid>, StoreError> {
        Ok(None)
    }

    async fn end(&self, _ctx: &RequestContext, _channel_id: &str) -> Result<u64, StoreError> {
        Ok(0)
    }

    async fn append(&self, _ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError> {
        if let Ok(mut kept) = self.turns.lock() {
            kept.extend_from_slice(turns);
        }
        Ok(())
    }
}

/// One stored conversation in the Rooms double.
struct Room {
    id: uuid::Uuid,
    tenant: String,
    user: String,
    channel: String,
    visibility: Visibility,
    ended: bool,
    touched: u64,
    turns: Vec<Message>,
}

/// A conversation store that keeps ownership, channels, visibility, turns, and ends the way
/// the database does.
#[derive(Default)]
pub struct Rooms {
    state: Mutex<(u64, Vec<Room>)>,
}

impl Rooms {
    fn with<T>(&self, f: impl FnOnce(&mut (u64, Vec<Room>)) -> T) -> Result<T, StoreError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| StoreError::Database("rooms poisoned".into()))?;
        Ok(f(&mut state))
    }
}

fn owned_by(room: &Room, ctx: &RequestContext) -> bool {
    room.tenant == ctx.tenant_id && room.user == ctx.user_id
}

#[async_trait]
impl ConversationStore for Rooms {
    async fn ensure(&self, ctx: &RequestContext, channel_id: &str) -> Result<(), StoreError> {
        self.with(|(clock, rooms)| {
            if let Some(room) = rooms.iter().find(|r| r.id == ctx.conversation_id) {
                let held = owned_by(room, ctx)
                    && room.channel == channel_id
                    && room.visibility == ctx.visibility;
                return if held {
                    Ok(())
                } else {
                    Err(StoreError::NotOwned)
                };
            }
            *clock += 1;
            rooms.push(Room {
                id: ctx.conversation_id,
                tenant: ctx.tenant_id.clone(),
                user: ctx.user_id.clone(),
                channel: channel_id.to_owned(),
                visibility: ctx.visibility,
                ended: false,
                touched: *clock,
                turns: Vec::new(),
            });
            Ok(())
        })?
    }

    async fn owns(&self, ctx: &RequestContext) -> Result<bool, StoreError> {
        self.with(|(_, rooms)| {
            rooms.iter().any(|r| {
                r.id == ctx.conversation_id && owned_by(r, ctx) && r.visibility == ctx.visibility
            })
        })
    }

    async fn load(&self, ctx: &RequestContext, limit: usize) -> Result<Vec<Message>, StoreError> {
        self.with(|(_, rooms)| {
            rooms
                .iter()
                .find(|r| r.id == ctx.conversation_id && owned_by(r, ctx))
                .map(|r| {
                    let skip = r.turns.len().saturating_sub(limit);
                    r.turns.iter().skip(skip).cloned().collect()
                })
                .unwrap_or_default()
        })
    }

    async fn append(&self, ctx: &RequestContext, turns: &[Message]) -> Result<(), StoreError> {
        self.with(|(clock, rooms)| {
            let room = rooms
                .iter_mut()
                .find(|r| r.id == ctx.conversation_id && owned_by(r, ctx))
                .ok_or(StoreError::NotOwned)?;
            *clock += 1;
            room.touched = *clock;
            room.turns.extend_from_slice(turns);
            Ok(())
        })?
    }

    async fn latest(
        &self,
        ctx: &RequestContext,
        channel_id: &str,
    ) -> Result<Option<uuid::Uuid>, StoreError> {
        self.with(|(_, rooms)| {
            rooms
                .iter()
                .filter(|r| {
                    owned_by(r, ctx)
                        && r.channel == channel_id
                        && r.visibility == ctx.visibility
                        && !r.ended
                })
                .max_by_key(|r| r.touched)
                .map(|r| r.id)
        })
    }

    async fn end(&self, ctx: &RequestContext, channel_id: &str) -> Result<u64, StoreError> {
        self.with(|(_, rooms)| {
            let mut ended = 0;
            for room in rooms
                .iter_mut()
                .filter(|r| owned_by(r, ctx) && r.channel == channel_id && !r.ended)
            {
                room.ended = true;
                ended += 1;
            }
            ended
        })
    }
}
