//! The sandbox: the flags that seal it, and what the tool hands back.

use std::sync::Arc;
use std::time::Duration;

use crate::core::config::SandboxSettings;
use crate::core::traits::tools::Tool;
use crate::core::traits::tools::sandbox::Sandbox;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::tools::sandbox::{SandboxError, SandboxOutput, SandboxRequest};
use crate::core::types::tools::{RiskClass, ToolError};
use crate::runtime::tools::sandbox::{ContainerSandbox, Limits, SandboxTool, Wording};

fn ctx() -> RequestContext {
    RequestContext::new("g", "u", Duration::from_secs(5))
}

#[test]
fn every_flag_that_seals_the_container_is_passed_once_with_its_value() {
    let args = ContainerSandbox::new(Limits::default()).args();

    // Each sealing flag appears exactly once, followed by its value.
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
    assert_eq!(
        args.last().map(String::as_str),
        Some(SandboxSettings::default().image.as_str())
    );
}

#[test]
fn the_workspace_is_writable_but_never_executable_and_is_sized_by_configuration() {
    let sealed = ContainerSandbox::new(Limits {
        workspace_mb: 128,
        ..Limits::default()
    })
    .seal();
    let at = sealed.iter().position(|a| a == "--tmpfs");
    let Some(at) = at else {
        unreachable!("the workspace is mounted")
    };
    assert_eq!(
        sealed.get(at + 1).map(String::as_str),
        Some("/tmp:rw,noexec,nosuid,size=128m")
    );
}

#[test]
fn a_long_output_keeps_its_head_and_its_tail() {
    use crate::core::types::tools::sandbox::workspace_path;
    use crate::runtime::tools::sandbox::clip;

    // The exit of a long run is at the end; head-only truncation would drop it.
    let long: String = std::iter::repeat_n('x', 100).collect();
    let text = format!("start{long}end");
    let out = clip(&text, 20);
    assert!(out.starts_with("start"), "{out}");
    assert!(out.ends_with("end"), "{out}");
    assert!(out.contains("characters cut"), "{out}");
    // Short output is untouched, and a zero limit keeps everything.
    assert_eq!(clip("short", 20), "short");
    assert_eq!(clip(&text, 0), text);

    // A workspace name cannot climb out of the workspace.
    assert!(workspace_path("result.txt").is_ok());
    for bad in ["../etc/passwd", "a/b", ".hidden", "", "a..b"] {
        assert!(workspace_path(bad).is_err(), "{bad} was accepted");
    }
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
    // The runtime does not exist; a Refused error shows it was never reached.
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

    async fn put(
        &self,
        _ctx: &RequestContext,
        _session: &str,
        name: &str,
        _content: &[u8],
    ) -> Result<String, SandboxError> {
        Ok(format!("/tmp/{name}"))
    }
}

/// The wording a test tool carries; the strings themselves are settings.
fn wording() -> Wording {
    Wording::from(&crate::core::config::SandboxSettings::default())
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
        wording(),
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
        wording(),
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
        wording(),
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

/// Live check against the container runtime: cargo test -p engine -- --ignored sandbox_runs.
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

    // The container has no network.
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

#[test]
fn a_caller_holding_the_most_sessions_loses_the_least_recently_used_one() {
    let s = ContainerSandbox::new(Limits {
        max_sessions: 2,
        ..Limits::default()
    });
    let ctx = ctx();
    let names: Vec<String> = ["first", "second"]
        .iter()
        .map(|n| ContainerSandbox::container_name(&ctx, n))
        .collect();
    for name in &names {
        s.note_used(name);
        // Distinct instants, so the oldest is unambiguous.
        std::thread::sleep(Duration::from_millis(2));
    }
    // Another caller's sessions never count against this one.
    let other = RequestContext::new("g", "someone-else", Duration::from_secs(5));
    s.note_used(&ContainerSandbox::container_name(&other, "theirs"));

    let evicted = s.least_recently_used(&ctx);
    assert_eq!(evicted.as_ref(), names.first());

    // Using the oldest again makes the other one the candidate.
    s.note_used(&names[0]);
    assert_eq!(s.least_recently_used(&ctx).as_ref(), names.get(1));
}

#[test]
fn a_session_idle_past_its_budget_is_swept_and_a_fresh_one_is_not() {
    let s = ContainerSandbox::new(Limits {
        session_idle_secs: 0,
        ..Limits::default()
    });
    let name = ContainerSandbox::container_name(&ctx(), "stale");
    s.note_used(&name);
    assert_eq!(s.idle_sessions(), vec![name]);

    let fresh = ContainerSandbox::new(Limits {
        session_idle_secs: 3_600,
        ..Limits::default()
    });
    fresh.note_used(&ContainerSandbox::container_name(&ctx(), "warm"));
    assert!(fresh.idle_sessions().is_empty());
}

/// Live check that the image carries what the tool description promises.
#[tokio::test]
#[ignore = "needs a container runtime"]
async fn the_image_carries_python_and_jq() {
    let s = ContainerSandbox::new(Limits {
        timeout: Duration::from_mins(1),
        ..Limits::default()
    });
    let ctx = RequestContext::new("g", "u", Duration::from_mins(1));

    for (command, want) in [
        (r"python3 -c 'print(6*7)'", "42"),
        (r#"echo '{"a":1}' | jq -r .a"#, "1"),
        (
            r"python3 -c 'import datetime;print((datetime.date(2026,1,2)-datetime.date(2026,1,1)).days)'",
            "1",
        ),
    ] {
        let Ok(out) = s
            .run(
                &ctx,
                &SandboxRequest {
                    command: command.into(),
                    session: None,
                },
            )
            .await
        else {
            unreachable!("the runtime answered")
        };
        assert_eq!(out.exit_code, 0, "{command}: {}", out.stderr);
        assert_eq!(out.stdout.trim(), want, "{command}");
    }
}

/// Live check that a result written to the workspace is there for the next command.
#[tokio::test]
#[ignore = "needs a container runtime"]
async fn a_put_file_is_read_back_by_the_next_command() {
    let s = ContainerSandbox::new(Limits {
        timeout: Duration::from_mins(1),
        ..Limits::default()
    });
    let ctx = RequestContext::new("g", "u", Duration::from_mins(1));
    let session = "putback";

    let Ok(path) = s
        .put(&ctx, session, "result.txt", b"alpha\nbeta\ngamma\n")
        .await
    else {
        unreachable!("the workspace took the file")
    };
    assert_eq!(path, "/tmp/result.txt");

    let Ok(out) = s
        .run(
            &ctx,
            &SandboxRequest {
                command: format!("grep -c . {path}"),
                session: Some(session.to_owned()),
            },
        )
        .await
    else {
        unreachable!("the runtime answered")
    };
    assert_eq!(out.stdout.trim(), "3", "{}", out.stderr);

    // A name that would climb out of the workspace is refused before anything runs.
    let escaped = s.put(&ctx, session, "../etc/passwd", b"x").await;
    assert!(
        matches!(escaped, Err(SandboxError::Refused(_))),
        "{escaped:?}"
    );
}

/// Live check that a session whose container was removed is started again, not failed.
#[tokio::test]
#[ignore = "needs a container runtime"]
async fn a_removed_session_is_started_again() {
    // A zero idle budget makes the sweep take every session it knows about.
    let s = ContainerSandbox::new(Limits {
        timeout: Duration::from_mins(1),
        session_idle_secs: 0,
        ..Limits::default()
    });
    let ctx = RequestContext::new("g", "u", Duration::from_mins(1));
    let session = Some("revive".to_owned());
    let request = |command: &str| SandboxRequest {
        command: command.to_owned(),
        session: session.clone(),
    };

    let Ok(first) = s.run(&ctx, &request("echo one > kept; echo ok")).await else {
        unreachable!("the runtime answered")
    };
    assert_eq!(first.exit_code, 0);

    // Reaping the session takes its workspace with it.
    s.reap_idle().await;

    let Ok(again) = s
        .run(&ctx, &request("cat kept 2>/dev/null; echo back"))
        .await
    else {
        unreachable!("a reaped session starts a new container rather than failing")
    };
    assert_eq!(again.exit_code, 0, "{}", again.stderr);
    assert_eq!(again.stdout.trim(), "back", "the workspace went with it");
}

#[test]
fn without_egress_the_container_has_no_network_and_no_proxy() {
    let sealed = ContainerSandbox::new(Limits::default()).seal();
    let at = sealed.iter().position(|a| a == "--network");
    assert_eq!(
        at.and_then(|i| sealed.get(i + 1)).map(String::as_str),
        Some("none")
    );
    assert!(
        !sealed
            .iter()
            .any(|a| a.contains("_PROXY") || a.contains("_proxy"))
    );
}

#[test]
fn with_egress_the_container_joins_the_internal_network_and_goes_out_through_the_proxy() {
    use crate::runtime::tools::sandbox::Egress;

    let sealed = ContainerSandbox::new(Limits {
        egress: Some(Egress {
            network: "sb-net".into(),
            proxy_image: "proxy:1".into(),
            proxy_name: "sb-proxy".into(),
        }),
        ..Limits::default()
    })
    .seal();
    let networks: Vec<&String> = sealed
        .iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "--network")
        .filter_map(|(i, _)| sealed.get(i + 1))
        .collect();
    assert_eq!(networks, [&"sb-net".to_owned()]);
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"] {
        let setting = format!("{var}=http://sb-proxy:3128");
        let count = sealed.iter().filter(|a| **a == setting).count();
        assert_eq!(count, 1, "{setting} in {sealed:?}");
    }
    // Everything else that seals the container stays in place.
    for flag in ["--read-only", "--cap-drop", "--security-opt", "--user"] {
        assert!(sealed.iter().any(|a| a == flag), "{flag} missing");
    }
}

/// Live check of egress: cargo test -p engine -- --ignored egress_reaches.
#[tokio::test]
#[ignore = "needs a container runtime and the sandbox and proxy images"]
async fn egress_reaches_the_public_web_and_nothing_private() {
    use crate::runtime::tools::sandbox::Egress;

    let s = ContainerSandbox::new(Limits {
        timeout: Duration::from_mins(1),
        egress: Some(Egress {
            network: "sparky-sandbox-test".into(),
            proxy_image: "ghcr.io/ashworks1706/sparkyai-sandbox-proxy:main".into(),
            proxy_name: "sparky-sandbox-proxy-test".into(),
        }),
        ..Limits::default()
    });
    assert!(s.prepare_egress().await.is_ok(), "egress is made ready");
    // A second boot finds everything in place.
    assert!(s.prepare_egress().await.is_ok(), "egress is ready again");
    let ctx = RequestContext::new("g", "u", Duration::from_mins(1));
    let run = |command: &str| SandboxRequest {
        command: command.into(),
        session: None,
    };
    let code = "curl -s -o /dev/null -w '%{http_code}' --max-time 20";

    let public = s
        .run(&ctx, &run(&format!("{code} https://www.asu.edu/")))
        .await;
    assert!(
        public
            .as_ref()
            .is_ok_and(|o| o.stdout.trim().starts_with('2') || o.stdout.trim().starts_with('3')),
        "{public:?}"
    );
    for private in [
        "http://169.254.169.254/",
        "http://172.17.0.1:5432/",
        "http://127.0.0.1/",
    ] {
        let out = s.run(&ctx, &run(&format!("{code} {private}"))).await;
        assert!(
            out.as_ref().is_ok_and(|o| o.stdout.trim() == "403"),
            "{private} is refused: {out:?}"
        );
    }
    let direct = s
        .run(
            &ctx,
            &run(&format!("{code} --noproxy '*' https://1.1.1.1/")),
        )
        .await;
    assert!(
        direct.as_ref().is_ok_and(|o| o.stdout.trim() == "000"),
        "no route around the proxy: {direct:?}"
    );
}
