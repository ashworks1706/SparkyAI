use std::collections::{HashMap, HashSet};

use crate::app::control::parse_command;
use crate::core::config::{Cli, repo_root};
use crate::core::types::{
    Command, Group, Kind, LogLine, SandboxCommand, ServiceState, Status, Stream,
};
use crate::units::catalog;
use crate::units::logs::{LogBuffer, LogWriter};
use crate::units::output::{parse_ps, sanitize_line};
use crate::units::sandbox::unwritten;

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
fn phoenix_is_an_infra_service_behind_its_profile() {
    let units = catalog();
    let phoenix = units.iter().find(|u| u.id == "phoenix");
    assert!(phoenix.is_some_and(|u| u.group == Group::Infra
        && u.url.as_deref() == Some("http://localhost:6006")
        && u.kind
            == Kind::Service {
                service: "phoenix".into(),
                profile: Some("phoenix".into()),
            }));
}

#[test]
fn cli_defaults_point_at_local_phoenix() {
    assert_eq!(Cli::default().phoenix_url, "http://localhost:6006");
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
fn log_writer_persists_unit_output_under_its_directory() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("sparky-cli-logs-{}", uuid::Uuid::new_v4()));
    let mut writer = LogWriter::new(&dir)?;

    writer.append("engine", &LogLine::now(Stream::Out, "ready"))?;

    let saved = std::fs::read_to_string(dir.join("engine.log"))?;
    assert!(saved.ends_with(" out ready\n"));
    std::fs::remove_dir_all(dir)?;
    Ok(())
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

#[test]
fn repo_root_is_found_from_a_nested_directory() {
    let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = repo_root(here).unwrap_or_default();
    assert!(root.join("justfile").is_file());
    assert!(repo_root(std::path::Path::new("/")).is_none());
}

#[test]
fn a_unit_url_reduces_to_the_address_a_port_check_can_reach() {
    use crate::units::health::address_of;

    assert_eq!(
        address_of("http://localhost:8080").as_deref(),
        Some("localhost:8080")
    );
    assert_eq!(
        address_of("localhost:5432").as_deref(),
        Some("localhost:5432")
    );
    assert_eq!(
        address_of("http://localhost:8000/v1").as_deref(),
        Some("localhost:8000")
    );
    // A unit bound to every interface is reached on loopback from here.
    assert_eq!(
        address_of("http://0.0.0.0:8080").as_deref(),
        Some("127.0.0.1:8080")
    );
    assert_eq!(address_of("http://localhost").as_deref(), None);
    assert_eq!(address_of("").as_deref(), None);
}

#[test]
fn a_port_already_listening_reads_as_served() {
    use std::time::Duration;

    use crate::units::health::served;

    let Ok(listener) = std::net::TcpListener::bind("127.0.0.1:0") else {
        unreachable!("a loopback port was free")
    };
    let Ok(addr) = listener.local_addr() else {
        unreachable!("a bound listener has an address")
    };
    let timeout = Duration::from_millis(500);
    assert!(served(&addr.to_string(), timeout), "{addr} is bound");

    drop(listener);
    assert!(
        !served(&format!("127.0.0.1:{}", addr.port()), timeout),
        "nothing listens once it is closed"
    );
}

fn command(id: u64, ended: bool) -> SandboxCommand {
    SandboxCommand {
        id,
        at: chrono::Utc::now(),
        container: Some("sparky-sb-00-scrape".into()),
        session: Some("scrape".into()),
        command: "python3 -c print(1)".into(),
        exit_code: ended.then_some(0),
        duration_ms: ended.then_some(12),
    }
}

#[test]
fn a_command_is_written_once_as_it_starts_and_once_with_how_it_went() {
    let mut shown: HashMap<u64, bool> = HashMap::new();

    // The engine reports it while it runs.
    let running = [command(1, false)];
    let first = unwritten(&running, &shown);
    assert_eq!(first.len(), 1);
    assert!(!first[0].ended());
    for c in &first {
        shown.insert(c.id, c.ended());
    }
    assert!(
        unwritten(&running, &shown).is_empty(),
        "the same report twice writes nothing twice"
    );

    // The next report carries the finished copy, which the old dedup on time would have lost.
    let done = [command(1, true)];
    let second = unwritten(&done, &shown);
    assert_eq!(second.len(), 1, "the end of a command is written");
    assert!(second[0].ended());
    for c in &second {
        shown.insert(c.id, c.ended());
    }
    assert!(unwritten(&done, &shown).is_empty());
}

#[test]
fn two_commands_sharing_a_moment_are_both_written() {
    let at = chrono::Utc::now();
    let both = [
        SandboxCommand {
            id: 2,
            at,
            ..command(2, true)
        },
        SandboxCommand {
            id: 3,
            at,
            ..command(3, true)
        },
    ];
    let written = unwritten(&both, &HashMap::new());
    assert_eq!(
        written.len(),
        2,
        "an id tells them apart, a timestamp does not"
    );
}

#[test]
fn a_command_says_whether_it_is_running_and_how_it_ended() {
    assert!(command(1, false).line().ends_with("-> running"));
    assert!(command(1, true).line().contains("exit 0 in 12ms"));
    let cancelled = SandboxCommand {
        exit_code: None,
        duration_ms: Some(9),
        ..command(1, true)
    };
    assert!(
        cancelled.line().contains("ended with no status"),
        "a cancelled command is not left reading as running: {}",
        cancelled.line()
    );
    assert!(cancelled.ended());
}
