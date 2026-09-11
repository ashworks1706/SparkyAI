//! The sandbox: the flags that seal it, and what the tool hands back.

use std::sync::Arc;
use std::time::Duration;

use crate::agent::tools::sandbox::{ContainerSandbox, Limits, SandboxTool};
use crate::core::config::SandboxSettings;
use crate::core::traits::sandbox::Sandbox;
use crate::core::traits::tool::Tool;
use crate::core::types::context::RequestContext;
use crate::core::types::sandbox::{SandboxError, SandboxOutput, SandboxRequest};
use crate::core::types::tool::{RiskClass, ToolError};

fn ctx() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5))
}

#[test]
fn every_flag_that_seals_the_container_is_passed() {
    let args = ContainerSandbox::new(Limits::default()).args();
    let joined = args.join(" ");
    // Each of these is the difference between a sandbox and a shell on the host.
    for flag in [
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--user 65534:65534",
        "--memory",
        "--cpus",
        "--pids-limit",
    ] {
        assert!(joined.contains(flag), "{flag} is missing from {joined}");
    }
    assert!(joined.contains("--rm"), "a container is not left behind");
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
    use crate::agent::tools::sandbox::ContainerSandbox;

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
    use crate::core::types::sandbox::session_name;

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
    use crate::agent::tools::sandbox::ContainerSandbox;

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
