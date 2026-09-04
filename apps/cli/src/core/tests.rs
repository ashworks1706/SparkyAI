use std::collections::HashSet;

use crate::app::parse_command;
use crate::core::config::repo_root;
use crate::core::types::{Command, Group, LogLine, ServiceState, Status, Stream};
use crate::logs::LogBuffer;
use crate::runner::{parse_ps, strip_ansi};
use crate::units::catalog;

#[test]
fn commands_parse_into_actions() {
    assert_eq!(parse_command("q"), Command::Quit);
    assert_eq!(
        parse_command("start engine"),
        Command::Start("engine".into())
    );
    assert_eq!(parse_command("stop  chat"), Command::Stop("chat".into()));
    assert_eq!(
        parse_command("restart discord"),
        Command::Restart("discord".into())
    );
    assert_eq!(
        parse_command("ask when does hayden close"),
        Command::Ask("when does hayden close".into())
    );
    assert_eq!(
        parse_command("roles mod, admin"),
        Command::Roles(vec!["mod".into(), "admin".into()])
    );
    assert_eq!(parse_command("reset"), Command::Reset);
    assert_eq!(parse_command("help"), Command::Help);
}

#[test]
fn unrecognised_words_become_just_recipes() {
    assert_eq!(
        parse_command("eval run"),
        Command::Just(vec!["eval".into(), "run".into()])
    );
    assert_eq!(
        parse_command("just check"),
        Command::Just(vec!["check".into()])
    );
    assert_eq!(parse_command("start"), Command::Just(vec!["start".into()]));
    assert_eq!(parse_command("   "), Command::Unknown(String::new()));
}

#[test]
fn catalog_ids_are_unique_and_grouped() {
    let units = catalog();
    let ids: HashSet<&str> = units.iter().map(|u| u.id.as_str()).collect();
    assert_eq!(ids.len(), units.len());
    assert!(units.iter().any(|u| u.id == "engine"));
    assert!(units.iter().any(|u| u.id == "eval run"));
    assert!(
        units
            .iter()
            .any(|u| u.id == "prod-up" && u.group == Group::Deploy)
    );
    assert!(
        units
            .iter()
            .all(|u| u.service().is_some() || !u.args.is_empty())
    );
}

#[test]
fn log_buffer_drops_oldest_and_searches_wrapping() {
    let mut b = LogBuffer::new(3);
    for t in ["alpha", "Beta", "gamma", "delta"] {
        b.push(LogLine::now(Stream::Out, t));
    }
    let texts: Vec<&str> = b.lines().map(|l| l.text.as_str()).collect();
    assert_eq!(texts, ["Beta", "gamma", "delta"]);
    assert_eq!(b.find("beta", 2, false), Some(0));
    assert_eq!(b.find("delta", 0, true), Some(2));
    assert_eq!(b.find("zeta", 0, false), None);
    assert_eq!(b.find("", 0, false), None);
}

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
    assert_eq!(parse_ps(array)["postgres"].health, "healthy");
    let lines = "{\"Service\":\"redis\",\"State\":\"exited\",\"ExitCode\":1}\n{\"Service\":\"minio\",\"State\":\"running\"}\n";
    let parsed = parse_ps(lines);
    assert_eq!(parsed["redis"].exit_code, 1);
    assert_eq!(parsed["minio"].status(), Status::Running);
    assert!(parse_ps("").is_empty());
}

#[test]
fn ansi_sequences_are_stripped() {
    assert_eq!(
        strip_ansi("\x1b[32m   Compiling\x1b[0m engine"),
        "   Compiling engine"
    );
    assert_eq!(strip_ansi("plain"), "plain");
}

#[test]
fn repo_root_is_found_from_a_nested_directory() {
    let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = repo_root(here).unwrap_or_default();
    assert!(root.join("justfile").is_file());
    assert!(repo_root(std::path::Path::new("/")).is_none());
}
