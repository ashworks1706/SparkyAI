//! Conversations on the platform: ownership, history, summaries, and channel endings.

use serde_json::json;
use uuid::Uuid;

use super::{Fake, MEMBER, caller, client, client_with};
use crate::core::traits::conversation::ConversationStore;
use crate::core::types::conversation::Visibility;
use crate::core::types::conversation::message::{Message, Role};
use crate::core::types::store::StoreError;
use crate::stores::platform::PlatformConversations;

#[tokio::test]
async fn ensure_sends_the_channel_and_visibility_and_a_conflict_is_not_owned() {
    let fake = Fake::default();
    let ctx = caller().with_visibility(Visibility::Private);
    let path = format!("{MEMBER}/conversations/{}", ctx.conversation_id);
    fake.on("PUT", &path, 200, json!({"id": ctx.conversation_id}));
    fake.on(
        "PUT",
        &path,
        409,
        json!({"error": "Conversation is not owned by this member"}),
    );
    let store = PlatformConversations::new(client(&fake).await);

    assert!(store.ensure(&ctx, "chan").await.is_ok());
    let sent = fake.last();
    assert_eq!(sent.method, "PUT");
    assert_eq!(sent.path, path);
    assert_eq!(
        sent.body,
        json!({"channel_id": "chan", "visibility": "private"})
    );
    assert!(matches!(
        store.ensure(&ctx, "chan").await,
        Err(StoreError::NotOwned)
    ));
}

#[tokio::test]
async fn owns_asks_at_the_request_visibility() {
    let fake = Fake::default();
    let ctx = caller();
    let path = format!("{MEMBER}/conversations/{}", ctx.conversation_id);
    fake.on("GET", &path, 200, json!({"owned": true}));
    let store = PlatformConversations::new(client(&fake).await);

    assert!(matches!(store.owns(&ctx).await, Ok(true)));
    assert_eq!(
        fake.last().query.get("visibility").map(String::as_str),
        Some("public")
    );
}

#[tokio::test]
async fn history_round_trips_whole_messages_and_summaries_keep_their_position() {
    let fake = Fake::default();
    let ctx = caller();
    let base = format!("{MEMBER}/conversations/{}", ctx.conversation_id);
    let summary = Message::summary("greeted");
    let hello = Message::assistant("hello");
    let (Ok(summary_json), Ok(hello_json)) =
        (serde_json::to_value(&summary), serde_json::to_value(&hello))
    else {
        unreachable!("messages serialize")
    };
    fake.on(
        "POST",
        &format!("{base}/messages"),
        201,
        json!({"seqs": [1, 2]}),
    );
    fake.on("POST", &format!("{base}/summary"), 201, json!({"seq": 3}));
    fake.on(
        "GET",
        &format!("{base}/messages"),
        200,
        json!({"messages": [
            {"position": 1, "role": "summary", "content": summary_json},
            {"position": 2, "role": "assistant", "content": hello_json},
        ]}),
    );
    let store = PlatformConversations::new(client(&fake).await);

    let turns = [Message::user("hi"), hello.clone()];
    assert!(store.append(&ctx, &turns).await.is_ok());
    let appended = fake.last().body;
    assert_eq!(appended["messages"][0]["role"], "user");
    assert_eq!(appended["messages"][0]["content"]["content"], "hi");
    assert_eq!(appended["messages"][1]["role"], "assistant");

    assert!(store.append_summary(&ctx, &summary, 1).await.is_ok());
    assert_eq!(fake.last().body["covers"], 1);
    assert_eq!(fake.last().body["content"]["role"], "summary");

    let Ok(loaded) = store.load(&ctx, 10).await else {
        unreachable!("history loads")
    };
    assert_eq!(
        fake.last().query.get("limit").map(String::as_str),
        Some("10")
    );
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].position, 1);
    assert_eq!(loaded[0].message.role, Role::Summary);
    assert_eq!(loaded[1].message, hello);
}

#[tokio::test]
async fn appending_nothing_sends_nothing() {
    let fake = Fake::default();
    let store = PlatformConversations::new(client(&fake).await);
    assert!(store.append(&caller(), &[]).await.is_ok());
    assert!(fake.seen().is_empty());
}

#[tokio::test]
async fn a_summary_role_appended_as_a_turn_is_stored_as_a_tool_row() {
    let fake = Fake::default();
    let ctx = caller();
    let path = format!("{MEMBER}/conversations/{}/messages", ctx.conversation_id);
    fake.on("POST", &path, 201, json!({"seqs": [1]}));
    let store = PlatformConversations::new(client(&fake).await);
    assert!(store.append(&ctx, &[Message::summary("s")]).await.is_ok());
    assert_eq!(fake.last().body["messages"][0]["role"], "tool");
}

#[tokio::test]
async fn latest_and_end_address_the_channel() {
    let fake = Fake::default();
    let ctx = caller();
    let id = Uuid::new_v4();
    fake.on(
        "GET",
        &format!("{MEMBER}/channels/c%201/latest"),
        200,
        json!({"id": id}),
    );
    fake.on(
        "GET",
        &format!("{MEMBER}/channels/empty/latest"),
        200,
        json!({"id": null}),
    );
    fake.on(
        "POST",
        &format!("{MEMBER}/channels/c%201/end"),
        200,
        json!({"ended": 2}),
    );
    let store = PlatformConversations::new(client(&fake).await);

    assert!(matches!(store.latest(&ctx, "c 1").await, Ok(Some(found)) if found == id));
    assert!(matches!(store.latest(&ctx, "empty").await, Ok(None)));
    assert!(matches!(store.end(&ctx, "c 1").await, Ok(2)));
}

#[tokio::test]
async fn a_wrong_token_is_an_error_not_an_empty_answer() {
    let fake = Fake::default();
    let store = PlatformConversations::new(client_with(&fake, "wrong").await);
    let error = store.owns(&caller()).await;
    assert!(matches!(&error, Err(StoreError::Database(m)) if m.contains("401")));
}
