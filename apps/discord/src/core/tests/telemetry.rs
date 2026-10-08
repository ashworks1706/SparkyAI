//! Span export to Phoenix and the resource it names.

use opentelemetry::Key;
use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
use secrecy::SecretString;

use crate::core::config::Telemetry;
use crate::core::telemetry::{phoenix_target, provider, resource};
use crate::core::tests::support::{Seen, serve, wait_for};

#[test]
fn discord_spans_reach_phoenix_without_a_token() {
    let seen = Seen::default();
    let addr = serve(std::sync::Arc::clone(&seen));
    assert!(addr.is_some());
    let Some(addr) = addr else {
        return;
    };
    let cfg = Telemetry {
        phoenix_url: Some(format!(" http://{addr}/ ")),
        ..Telemetry::default()
    };
    assert_eq!(
        phoenix_target(&cfg),
        Some(format!("http://{addr}/v1/traces"))
    );
    let built = provider(&cfg, "discord-test", "test");
    assert!(built.as_ref().is_ok_and(Option::is_some), "{built:?}");
    let Ok(Some(provider)) = built else {
        return;
    };
    let mut span = provider.tracer("discord-test").start("probe");
    span.end();
    let flushed = provider.force_flush();
    assert!(flushed.is_ok(), "{flushed:?}");

    let got = wait_for(&seen, 1);
    let _ = provider.shutdown();
    assert_eq!(
        got.iter().map(|(p, _, _)| p.as_str()).collect::<Vec<_>>(),
        ["/v1/traces"]
    );
    assert!(got.iter().all(|(_, auth, _)| auth.is_empty()), "{got:?}");
}

#[test]
fn a_phoenix_api_key_is_sent_as_a_bearer_token() {
    let seen = Seen::default();
    let addr = serve(std::sync::Arc::clone(&seen));
    assert!(addr.is_some());
    let Some(addr) = addr else {
        return;
    };
    let cfg = Telemetry {
        phoenix_url: Some(format!("http://{addr}")),
        phoenix_api_key: SecretString::from("px_test".to_owned()),
        ..Telemetry::default()
    };
    let built = provider(&cfg, "discord-test", "test");
    let Ok(Some(provider)) = built else {
        unreachable!("a set phoenix_url builds a provider")
    };
    let mut span = provider.tracer("discord-test").start("probe");
    span.end();
    let _ = provider.force_flush();

    let got = wait_for(&seen, 1);
    let _ = provider.shutdown();
    assert!(
        got.iter().all(|(_, auth, _)| auth == "Bearer px_test"),
        "{got:?}"
    );
}

#[test]
fn export_is_off_until_phoenix_is_set() {
    assert!(matches!(
        provider(&Telemetry::default(), "d", "test"),
        Ok(None)
    ));
    let phoenix = Telemetry {
        phoenix_url: Some("http://localhost:6006".into()),
        ..Telemetry::default()
    };
    assert!(matches!(provider(&phoenix, "d", "test"), Ok(Some(_))));
}

#[test]
fn the_discord_resource_names_the_phoenix_project() {
    let cfg = Telemetry {
        project_name: "sparky-test".into(),
        ..Telemetry::default()
    };
    let resource = resource(&cfg, "discord-test", "test");
    assert_eq!(
        resource
            .get(&Key::from_static_str("openinference.project.name"))
            .map(|v| v.to_string()),
        Some("sparky-test".to_owned())
    );
}
