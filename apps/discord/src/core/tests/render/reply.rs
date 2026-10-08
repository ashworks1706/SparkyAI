//! Reply text: chunking, failure wording, and the memory listing.

use crate::core::types::profile::{ProfileNode, ProfileRelation};
use crate::core::types::{EngineError, ProfileList};
use crate::render::reply::{
    MAX_MESSAGE, NOT_YOURS, NOTHING_REMEMBERED, UNAVAILABLE, chunk, confirm_failure,
    confirm_refused, failure, forgot, memory_failure, render_profile,
};

#[test]
fn short_text_is_one_message() {
    assert_eq!(chunk("hello", MAX_MESSAGE), vec!["hello".to_owned()]);
}

#[test]
fn long_text_splits_on_line_boundaries_under_the_limit() {
    let text = (0..200)
        .map(|i| format!("line {i} {}", "x".repeat(40)))
        .collect::<Vec<_>>()
        .join("\n");
    let parts = chunk(&text, 500);
    assert!(parts.len() > 1);
    assert!(parts.iter().all(|p| p.len() <= 500));
    assert!(parts.iter().all(|p| !p.ends_with('\n')));
    let rejoined = parts.join("\n");
    assert!(rejoined.contains("line 199"));
}

#[test]
fn chunking_always_advances_even_at_tiny_limits() {
    let parts = chunk("ééé", 1);
    assert_eq!(parts, vec!["é", "é", "é"]);
    let parts = chunk("a é b", 2);
    assert_eq!(parts.concat().replace(' ', ""), "aéb");
}

#[test]
fn capacity_and_outage_read_differently_to_the_user() {
    let busy = failure(&EngineError::Status {
        status: 503,
        body: "the model is at capacity".into(),
    });
    assert!(busy.contains("busy"), "{busy}");

    let down = failure(&EngineError::Transport("connection refused".into()));
    assert!(down.contains("unavailable"), "{down}");

    let broken = failure(&EngineError::Status {
        status: 502,
        body: String::new(),
    });
    assert!(broken.contains("unavailable"), "{broken}");

    let gone = failure(&EngineError::Status {
        status: 404,
        body: "no such conversation".into(),
    });
    assert!(gone.contains("Ask again"), "{gone}");
}

#[test]
fn a_refused_press_is_told_privately_and_other_failures_read_as_outages() {
    let status = |status| EngineError::Status {
        status,
        body: String::new(),
    };
    assert!(confirm_refused(&status(404)));
    assert!(confirm_refused(&status(409)));
    assert!(!confirm_refused(&status(502)));
    assert!(!confirm_refused(&EngineError::Transport("down".into())));
    assert_eq!(confirm_failure(&status(404)), NOT_YOURS);
    assert_eq!(confirm_failure(&status(500)), UNAVAILABLE);
    assert_eq!(failure(&status(409)), "That approval is no longer open.");
}

#[test]
fn memory_renders_things_and_relations_or_says_it_is_empty() {
    assert_eq!(
        render_profile(&ProfileList::default(), 2_000),
        vec![NOTHING_REMEMBERED.to_owned()]
    );

    let profile = ProfileList {
        nodes: vec![ProfileNode {
            kind: "course".into(),
            label: "CSE 310".into(),
            confidence: 0.87,
        }],
        relations: vec![ProfileRelation {
            subject: "me".into(),
            relation: "enrolled_in".into(),
            object: "CSE 310".into(),
            confidence: 0.5,
        }],
    };
    let out = render_profile(&profile, 2_000).join("\n");
    assert!(out.contains("- CSE 310 (course, 87%)"), "{out}");
    assert!(out.contains("- me enrolled in CSE 310 (50%)"), "{out}");

    let many = ProfileList {
        nodes: (0..200)
            .map(|i| ProfileNode {
                kind: "club".into(),
                label: format!("club number {i}"),
                confidence: 1.0,
            })
            .collect(),
        relations: vec![],
    };
    let parts = render_profile(&many, 500);
    assert!(parts.len() > 1);
    assert!(parts.iter().all(|p| p.len() <= 500));

    assert_eq!(forgot(0, true), "I had nothing under that name.");
    assert_eq!(forgot(3, false), "Forgot 3 things.");
    let off = memory_failure(&EngineError::Status {
        status: 503,
        body: String::new(),
    });
    assert_eq!(off, "Memory is turned off here.");
}
