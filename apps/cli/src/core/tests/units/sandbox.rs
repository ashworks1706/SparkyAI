//! The sandbox command log.

use std::collections::HashMap;

use crate::core::types::SandboxCommand;
use crate::units::sandbox::unwritten;

/// A sandbox command with id, ended or still running.
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
    assert_eq!(
        unwritten(&done, &shown).len(),
        0,
        "nothing is written twice"
    );
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
