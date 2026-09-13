//! A public request loads no personal memory; a private one does and says which memories it used.

use std::sync::Arc;

use crate::core::tests::support::{Known, Recalling, Scripted, agent_recalling, ctx, text};
use crate::core::types::agent::AgentConfig;
use crate::core::types::conversation::Visibility;

/// What one turn recalled: memory store calls, profile graph calls, memories the answer reports.
async fn recalls(visibility: Visibility, recall_in_public: bool) -> (usize, usize, Vec<String>) {
    let memory = Arc::new(Recalling::default());
    let graph = Arc::new(Known::default());
    let cfg = AgentConfig {
        recall_in_public,
        ..AgentConfig::default()
    };
    let agent = agent_recalling(
        Scripted::new(vec![Ok(text("hi"))]),
        cfg,
        memory.clone(),
        graph.clone(),
    );
    let answered = agent
        .run(&ctx().with_visibility(visibility), "what do I study")
        .await;
    let memories = answered.map(|a| a.memories).unwrap_or_default();
    (memory.calls(), graph.recalls(), memories)
}

#[tokio::test]
async fn a_public_request_recalls_no_memory_and_no_profile() {
    let (memory, profile, memories) = recalls(Visibility::Public, false).await;
    assert_eq!((memory, profile), (0, 0));
    assert!(memories.is_empty());
}

#[tokio::test]
async fn a_private_request_recalls_memory_and_profile_and_reports_them() {
    let (memory, profile, memories) = recalls(Visibility::Private, false).await;
    assert_eq!((memory, profile), (1, 1));
    assert_eq!(
        memories,
        vec![
            "studies CSE 310".to_owned(),
            "course: CSE 310".to_owned(),
            "the user studies CSE 310".to_owned(),
        ]
    );
}

#[tokio::test]
async fn the_setting_lets_a_public_request_recall_but_never_name_what_it_recalled() {
    let (memory, profile, memories) = recalls(Visibility::Public, true).await;
    assert_eq!((memory, profile), (1, 1));
    assert!(memories.is_empty());
}

#[test]
fn a_context_starts_public() {
    assert_eq!(ctx().visibility, Visibility::Public);
}
