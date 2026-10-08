//! The platform section: what it needs when on, and what it rules out.

use figment::Figment;
use figment::providers::{Format, Toml};

use crate::core::config::Config;

/// The sections with no defaults, without a PostgreSQL URL.
const BASE: &str = r#"
[app]
env = "test"
http_addr = "127.0.0.1:0"
log_level = "info"
[engine]
service_token = "t"
[discord]
guild_id = 1
[model]
base_url = "http://localhost:8000/v1"
api_key = ""
name = "m"
max_tokens = 256
[embedding]
base_url = "http://localhost:8001/v1"
api_key = ""
name = "e"
dim = 1024
"#;

/// A platform section that is on and complete.
const ON: &str = r#"
[platform]
enabled = true
url = "https://platform.example.org"
token = "plat_x"
"#;

fn load(extra: &str) -> Result<Config, String> {
    let cfg: Config = Figment::new()
        .merge(Toml::string(BASE))
        .merge(Toml::string(extra))
        .extract()
        .map_err(|e| e.to_string())?;
    cfg.validate().map_err(|e| e.to_string())?;
    Ok(cfg)
}

#[cfg(feature = "standalone")]
#[test]
fn platform_is_off_by_default_and_then_postgres_is_required() {
    assert!(load("").is_err_and(|e| e.contains("SPARKY_POSTGRES__URL")));
    let cfg =
        load("[postgres]\nurl = \"postgres://localhost/x\"\n[query.cache]\nenabled = false\n");
    assert!(cfg.is_ok_and(|c| !c.platform.enabled));
}

#[cfg(not(feature = "standalone"))]
#[test]
fn a_build_without_standalone_refuses_platform_off() {
    let with_postgres = load("[postgres]\nurl = \"postgres://localhost/x\"\n");
    assert!(with_postgres.is_err_and(|e| e.contains("standalone")));
}

#[test]
fn platform_on_boots_without_postgres_or_redis() {
    let Ok(cfg) = load(ON) else {
        unreachable!("a complete platform section is valid")
    };
    assert!(cfg.postgres.url.is_none());
    assert!(cfg.redis.is_none());
    assert_eq!(cfg.platform.timeout_secs, 10);
    assert_eq!(cfg.platform.max_query_chars, 1000);
    assert_eq!(cfg.platform.embedding_model, "");
}

#[test]
fn platform_on_needs_a_url_and_a_token() {
    let no_url = "[platform]\nenabled = true\ntoken = \"plat_x\"\n";
    assert!(load(no_url).is_err_and(|e| e.contains("SPARKY_PLATFORM__URL")));
    let bad_url = "[platform]\nenabled = true\nurl = \"ftp://x\"\ntoken = \"plat_x\"\n";
    assert!(load(bad_url).is_err_and(|e| e.contains("SPARKY_PLATFORM__URL")));
    let no_token = "[platform]\nenabled = true\nurl = \"https://p.example\"\n";
    assert!(load(no_token).is_err_and(|e| e.contains("SPARKY_PLATFORM__TOKEN")));
}

#[test]
fn platform_on_rejects_what_the_platform_cannot_serve() {
    let with = |more: &str| load(&format!("{ON}{more}"));
    assert!(with("timeout_secs = 0\n").is_err());
    assert!(with("max_query_chars = 1001\n").is_err());
    assert!(with("[retrieval]\ntop_k = 51\n").is_err_and(|e| e.contains("top_k")));
    assert!(with("[retrieval]\nwindow = 6\n").is_err_and(|e| e.contains("window")));
    assert!(
        with("[agent]\nconfirmation_ttl_secs = 90000\n")
            .is_err_and(|e| e.contains("confirmation_ttl_secs"))
    );
}

#[test]
fn platform_on_runs_the_login_so_engine_oauth_clients_are_refused() {
    let canvas = r#"
[oauth.canvas]
enabled = true
client_id = "id"
client_secret = "secret"
redirect_url = "https://engine.example/oauth/canvas/callback"
authorize_url = "https://canvas.example/login/oauth2/auth"
token_url = "https://canvas.example/login/oauth2/token"
"#;
    assert!(load(&format!("{ON}{canvas}")).is_err_and(|e| e.contains("oauth.canvas")));
}

#[test]
fn platform_on_needs_no_redis_for_the_query_cache() {
    let cached = "[query.cache]\nenabled = true\n";
    assert!(load(&format!("{ON}{cached}")).is_ok());
}
