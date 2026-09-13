//! The sandbox: the flags that seal it, and what the tool hands back.

use std::sync::Arc;
use std::time::Duration;

use crate::core::config::SandboxSettings;
use crate::core::traits::tools::Tool;
use crate::core::traits::tools::sandbox::Sandbox;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::sandbox::{SandboxError, SandboxOutput, SandboxRequest};
use crate::core::types::tools::{RiskClass, ToolError};
use crate::runtime::tools::sandbox::{ContainerSandbox, Limits, SandboxTool};

fn ctx() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5))
}

#[test]
fn every_flag_that_seals_the_container_is_passed_once_with_its_value() {
    let args = ContainerSandbox::new(Limits::default()).args();

    // Each of these is the difference between a sandbox and a shell on the host. Checking the
    // joined string for a substring is not enough: a flag repeated by accident still contains
    // the pair, and docker then reads the value as the image name.
    for (flag, value) in [
        ("--network", Some("none")),
        ("--read-only", None),
        ("--cap-drop", Some("ALL")),
        ("--security-opt", Some("no-new-privileges")),
        ("--user", Some("65534:65534")),
        ("--memory", Some("256m")),
        ("--cpus", Some("1")),
        ("--pids-limit", Some("128")),
        ("--rm", None),
    ] {
        let at: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| a.as_str() == flag)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(at.len(), 1, "{flag} appears {} times in {args:?}", at.len());
        if let Some(value) = value {
            assert_eq!(args.get(at[0] + 1).map(String::as_str), Some(value));
        }
    }

    // The image is the last argument, and the command follows it.
    assert_eq!(args.last().map(String::as_str), Some("alpine:3.20"));
}

#[test]
fn the_limits_come_from_configuration() {
    let cfg = SandboxSettings::default();
    let l = Limits::from(&cfg);
    assert_eq!(l.runtime, cfg.runtime);
    assert_eq!(l.image, cfg.image);
    assert_eq!(l.memory, cfg.memory);
    assert_eq!(l.timeout.as_secs(), cfg.timeout_secs);
    assert!(
        !cfg.enabled,
        "a sandbox is offered only when a deployment asks for it"
    );
}

#[tokio::test]
async fn an_empty_command_is_refused_before_a_container_starts() {
    let s = ContainerSandbox::new(Limits {
        runtime: "definitely-not-a-real-binary".into(),
        ..Limits::default()
    });
    let out = s
        .run(
            &ctx(),
            &SandboxRequest {
                command: "   ".into(),
                session: None,
            },
        )
        .await;
    // The runtime does not exist, so reaching it would be a Runtime error instead.
    assert!(matches!(out, Err(SandboxError::Refused(_))), "{out:?}");
}

#[tokio::test]
async fn a_missing_runtime_is_reported_rather_than_hidden() {
    let s = ContainerSandbox::new(Limits {
        runtime: "definitely-not-a-real-binary".into(),
        ..Limits::default()
    });
    let out = s
        .run(
            &ctx(),
            &SandboxRequest {
                command: "echo hi".into(),
                session: None,
            },
        )
        .await;
    assert!(matches!(out, Err(SandboxError::Runtime(_))), "{out:?}");
}

/// A sandbox that answers without running anything.
struct Canned(SandboxOutput);

#[async_trait::async_trait]
impl Sandbox for Canned {
    async fn run(
        &self,
        _ctx: &RequestContext,
        _request: &SandboxRequest,
    ) -> Result<SandboxOutput, SandboxError> {
        Ok(self.0.clone())
    }
}

#[tokio::test]
async fn a_non_zero_exit_is_reported_and_is_not_a_tool_failure() {
    let tool = SandboxTool::new(
        Arc::new(Canned(SandboxOutput {
            exit_code: 2,
            stdout: String::new(),
            stderr: "no such file".into(),
            session: None,
        })),
        RiskClass::PrepareWrite,
    );
    let Ok(out) = tool
        .call(&ctx(), serde_json::json!({"command": "cat missing"}))
        .await
    else {
        unreachable!("a failing command is a result the model reads, not a failed tool")
    };
    assert!(out.content.contains("exit 2"), "{}", out.content);
    assert!(out.content.contains("no such file"), "{}", out.content);
}

#[tokio::test]
async fn the_declared_risk_is_what_policy_gates_it_by() {
    let tool = SandboxTool::new(
        Arc::new(Canned(SandboxOutput {
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
            session: None,
        })),
        RiskClass::Destructive,
    );
    assert_eq!(tool.definition().risk, RiskClass::Destructive);
    assert!(tool.definition().sequential, "one container at a time");
}

#[tokio::test]
async fn arguments_that_do_not_carry_a_command_are_a_correctable_refusal() {
    let tool = SandboxTool::new(
        Arc::new(Canned(SandboxOutput {
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
            session: None,
        })),
        RiskClass::PrepareWrite,
    );
    let out = tool.call(&ctx(), serde_json::json!({"cmd": "echo"})).await;
    assert!(
        matches!(out, Err(ToolError::InvalidArguments(_))),
        "{out:?}"
    );
}

#[test]
fn a_session_container_is_scoped_to_the_caller() {
    use crate::runtime::tools::sandbox::ContainerSandbox;

    let mine = RequestContext::new("guild", "me", Duration::from_secs(5));
    let yours = RequestContext::new("guild", "you", Duration::from_secs(5));
    let other_guild = RequestContext::new("elsewhere", "me", Duration::from_secs(5));

    // Naming the same session from another account must not reach the container of the first.
    assert_ne!(
        ContainerSandbox::container_name(&mine, "work"),
        ContainerSandbox::container_name(&yours, "work")
    );
    assert_ne!(
        ContainerSandbox::container_name(&mine, "work"),
        ContainerSandbox::container_name(&other_guild, "work")
    );
    // The same caller naming the same session reaches the same container, which is resuming.
    assert_eq!(
        ContainerSandbox::container_name(&mine, "work"),
        ContainerSandbox::container_name(&mine, "work")
    );
}

#[test]
fn a_session_name_that_could_reach_another_container_is_refused() {
    use crate::core::types::tools::sandbox::session_name;

    for bad in [
        "",
        "   ",
        "has space",
        "../escape",
        "semi;colon",
        "$(sub)",
        &"x".repeat(49),
    ] {
        assert!(session_name(bad).is_err(), "{bad:?} was accepted");
    }
    for good in ["work", "run-1", "my_session", "A1"] {
        assert!(session_name(good).is_ok(), "{good:?} was refused");
    }
}

#[test]
fn a_session_call_keeps_the_container_and_a_plain_call_does_not() {
    use crate::runtime::tools::sandbox::ContainerSandbox;

    let s = ContainerSandbox::new(Limits::default());
    // Without a session the container is removed when the command exits.
    assert!(s.args().contains(&"--rm".to_owned()));
    // The sealing flags are the same on both paths.
    let seal = s.seal().join(" ");
    assert!(seal.contains("--network none"), "{seal}");
    assert!(seal.contains("--read-only"), "{seal}");
    assert!(seal.contains("--cap-drop ALL"), "{seal}");
    assert!(
        !seal.contains("--rm"),
        "a session container outlives one command"
    );
}

/// Live check against the container runtime. Run it with
/// cargo test -p engine -- --ignored sandbox_runs.
#[tokio::test]
#[ignore = "needs a container runtime"]
async fn a_real_container_computes_and_stays_sealed() {
    let s = ContainerSandbox::new(Limits {
        timeout: Duration::from_mins(1),
        ..Limits::default()
    });
    let ctx = RequestContext::new("g", "u", Duration::from_mins(1));

    let Ok(out) = s
        .run(
            &ctx,
            &SandboxRequest {
                command: "echo $((7*191))".into(),
                session: None,
            },
        )
        .await
    else {
        unreachable!("the runtime answered")
    };
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.trim(), "1337");

    // The seal is the reason this tool is allowed to exist at all.
    let Ok(net) = s
        .run(
            &ctx,
            &SandboxRequest {
                command: "wget -T2 -q -O- https://example.com".into(),
                session: None,
            },
        )
        .await
    else {
        unreachable!("the runtime answered")
    };
    assert_ne!(net.exit_code, 0, "the network is refused");

    let Ok(fs) = s
        .run(
            &ctx,
            &SandboxRequest {
                command: "touch /etc/x".into(),
                session: None,
            },
        )
        .await
    else {
        unreachable!("the runtime answered")
    };
    assert_ne!(fs.exit_code, 0, "the root filesystem is read only");
}

/// Live check that a session keeps what an earlier command wrote.
#[tokio::test]
#[ignore = "needs a container runtime"]
async fn a_real_session_keeps_state_between_calls() {
    let s = ContainerSandbox::new(Limits {
        timeout: Duration::from_mins(1),
        session_idle_secs: 60,
        ..Limits::default()
    });
    let ctx = RequestContext::new("g", "u", Duration::from_mins(1));
    let session = Some("itest".to_owned());

    let first = s
        .run(
            &ctx,
            &SandboxRequest {
                command: "echo kept > state.txt".into(),
                session: session.clone(),
            },
        )
        .await;
    assert!(first.is_ok(), "{first:?}");

    let Ok(second) = s
        .run(
            &ctx,
            &SandboxRequest {
                command: "cat state.txt".into(),
                session,
            },
        )
        .await
    else {
        unreachable!("the session resumed")
    };
    assert_eq!(second.stdout.trim(), "kept");
}
