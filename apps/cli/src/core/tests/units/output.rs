//! Child output: compose rows, their statuses, and sanitized lines.

use crate::core::types::{ServiceState, Status};
use crate::units::output::{parse_ps, sanitize_line};

#[test]
fn compose_states_map_onto_statuses() {
    let running = ServiceState {
        state: "running".into(),
        health: "healthy".into(),
        exit_code: 0,
    };
    assert_eq!(running.status(), Status::Running);
    let starting = ServiceState {
        state: "running".into(),
        health: "starting".into(),
        exit_code: 0,
    };
    assert_eq!(starting.status(), Status::Starting);
    let sick = ServiceState {
        state: "running".into(),
        health: "unhealthy".into(),
        exit_code: 0,
    };
    assert_eq!(sick.status(), Status::Failed("unhealthy".into()));
    let stopped = ServiceState {
        state: "exited".into(),
        health: String::new(),
        exit_code: 0,
    };
    assert_eq!(stopped.status(), Status::Stopped);
    let dead = ServiceState {
        state: "exited".into(),
        health: String::new(),
        exit_code: 137,
    };
    assert_eq!(dead.status(), Status::Exited(137));
}

#[test]
fn ps_output_parses_as_array_or_lines() {
    let array = r#"[{"Service":"postgres","State":"running","Health":"healthy","ExitCode":0}]"#;
    assert_eq!(
        parse_ps(array).unwrap_or_default()["postgres"].health,
        "healthy"
    );
    let lines = "{\"Service\":\"redis\",\"State\":\"exited\",\"ExitCode\":1}\n{\"Service\":\"minio\",\"State\":\"running\"}\n";
    let parsed = parse_ps(lines).unwrap_or_default();
    assert_eq!(parsed["redis"].exit_code, 1);
    assert_eq!(parsed["minio"].status(), Status::Running);
    assert!(parse_ps("").is_ok_and(|m| m.is_empty()));
    assert!(parse_ps("not json").is_err());
}

#[test]
fn a_log_line_cannot_move_the_cursor() {
    assert_eq!(
        sanitize_line("\x1b[32m   Compiling\x1b[0m engine"),
        "   Compiling engine"
    );
    assert_eq!(sanitize_line("plain"), "plain");
    // docker compose rewrites its progress lines in place with a carriage return.
    assert_eq!(
        sanitize_line("Container deploy-embed-1  Recreated\r"),
        "Container deploy-embed-1  Recreated"
    );
    assert_eq!(sanitize_line("a\rb\x08c\x07"), "abc");
    assert_eq!(sanitize_line("keeps\ttabs"), "keeps\ttabs");
}
