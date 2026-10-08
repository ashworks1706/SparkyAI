//! Memories on the platform.

use serde_json::json;

use super::{Fake, MEMBER, caller, client};
use crate::core::traits::memory::MemoryStore;
use crate::core::types::memory::{MemoryKind, MemoryQuery};
use crate::stores::platform::PlatformMemory;

#[tokio::test]
async fn recall_names_the_kinds_and_limit_and_reads_platform_rows() {
    let fake = Fake::default();
    fake.on(
        "GET",
        &format!("{MEMBER}/memories"),
        200,
        json!({"memories": [{
            "id": "6b1f6d2e-6c1e-4d4e-9a55-0a5c3c2f1b11",
            "kind": "semantic",
            "content": "Takes CSE 310",
            "sensitivity": "normal",
            "confidence": 0.9,
            "source_seq": null,
            "created_at": "2026-10-08T01:00:00.123456",
            "expires_at": "2026-11-08T01:00:00+00:00"
        }]}),
    );
    let query = MemoryQuery {
        kinds: vec![MemoryKind::Semantic, MemoryKind::Task],
        limit: 5,
    };
    let Ok(recalled) = PlatformMemory::new(client(&fake).await)
        .recall(&caller(), &query)
        .await
    else {
        unreachable!("memories load")
    };
    let sent = fake.last();
    assert_eq!(
        sent.query.get("kinds").map(String::as_str),
        Some("semantic,task")
    );
    assert_eq!(sent.query.get("limit").map(String::as_str), Some("5"));
    assert_eq!(recalled.len(), 1);
    assert_eq!(recalled[0].content, "Takes CSE 310");
    assert_eq!(recalled[0].kind, MemoryKind::Semantic);
    assert!(recalled[0].expires_at.is_some());
}

#[tokio::test]
async fn an_unknown_kind_is_an_error() {
    let fake = Fake::default();
    fake.on(
        "GET",
        &format!("{MEMBER}/memories"),
        200,
        json!({"memories": [{
            "id": "6b1f6d2e-6c1e-4d4e-9a55-0a5c3c2f1b11",
            "kind": "dream",
            "content": "x",
            "confidence": 1.0,
            "created_at": "2026-10-08T01:00:00",
            "expires_at": null
        }]}),
    );
    let query = MemoryQuery {
        kinds: Vec::new(),
        limit: 5,
    };
    let result = PlatformMemory::new(client(&fake).await)
        .recall(&caller(), &query)
        .await;
    assert!(result.is_err());
    assert!(!fake.last().query.contains_key("kinds"));
}
