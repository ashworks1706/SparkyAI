//! Configuration defaults, layering, and validation.

use crate::core::config::{Agent, Config, Trace};

#[test]
fn traces_default_to_the_sparky_state_directory() {
    assert_eq!(Trace::default().dir, ".sparky/traces");
}

#[test]
fn every_section_budget_fits_inside_the_prompt_budget() {
    let a = Agent::default();
    for section in [
        a.evidence_budget_tokens,
        a.history_budget_tokens,
        a.memory_budget_tokens,
    ] {
        assert!(section <= a.prompt_budget_tokens);
    }
}

/// Loads a config from the sections that have no defaults plus `extra`, so what is under test
/// is `validate` and the defaults rather than deserialization.
fn load(extra: &str) -> Result<Config, String> {
    use figment::providers::{Format, Toml};

    let base = r#"
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
[postgres]
url = "postgres://localhost/x"
[embedding]
base_url = "http://localhost:8001/v1"
api_key = ""
name = "e"
dim = 1024
"#;
    let cfg: Config = figment::Figment::new()
        .merge(Toml::string(base))
        .merge(Toml::string(extra))
        .extract()
        .map_err(|e| e.to_string())?;
    cfg.validate().map_err(|e| e.to_string())?;
    Ok(cfg)
}

/// The config `extra` produces, or a failure naming why it did not load.
fn ok(extra: &str) -> Config {
    match load(extra) {
        Ok(cfg) => cfg,
        Err(e) => unreachable!("expected a valid config, got {e}"),
    }
}

/// The rejection `extra` produces, or a failure saying it was accepted.
fn err(extra: &str) -> String {
    match load(extra) {
        Ok(_) => unreachable!("expected {extra:?} to be rejected"),
        Err(e) => e,
    }
}

#[test]
fn a_bare_config_is_valid_and_takes_every_default() {
    let cfg = ok("");
    assert_eq!(cfg.retrieval.top_k, 6);
    assert_eq!(cfg.agent.chars_per_token, 4);
    assert!(cfg.trace.enabled);
    assert_eq!(cfg.policy.write_roles, vec!["MANAGE_GUILD".to_owned()]);
}

#[test]
fn turning_off_both_retrieval_legs_is_rejected() {
    // With neither leg the engine would answer every question with no evidence at all.
    let e = err("[retrieval]\ndense = false\nlexical = false\n");
    assert!(e.contains("both off"), "{e}");
}

#[test]
fn a_section_budget_above_the_prompt_budget_is_rejected() {
    let e = err("[agent]\nprompt_budget_tokens = 1000\nevidence_budget_tokens = 4000\n");
    assert!(e.contains("evidence_budget_tokens"), "{e}");
}

#[test]
fn a_text_search_configuration_that_is_not_an_identifier_is_rejected() {
    // It is interpolated into SQL, so nothing but a plain identifier may reach the database.
    let e = err("[retrieval]\ntext_search_config = \"english'; drop table chunks --\"\n");
    assert!(e.contains("text_search_config"), "{e}");
}

#[test]
fn a_sample_ratio_outside_zero_to_one_is_rejected() {
    let e = err("[telemetry]\nsample_ratio = 2.0\n");
    assert!(e.contains("sample_ratio"), "{e}");
}

#[test]
fn the_legacy_playwright_url_becomes_a_named_server() {
    let cfg = ok("[mcp]\nplaywright_url = \"http://localhost:8931/mcp\"\n");
    let servers = cfg.mcp.resolved_servers();
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0].name, "playwright");
    assert!(servers[0].tools.contains(&"browser_snapshot".to_owned()));
}

#[test]
fn a_server_list_replaces_the_legacy_entry_of_the_same_name() {
    let cfg = ok("[mcp]\nplaywright_url = \"http://legacy/mcp\"\n\
                  [[mcp.servers]]\nname = \"playwright\"\nurl = \"http://new/mcp\"\n");
    let servers = cfg.mcp.resolved_servers();
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0].url, "http://new/mcp");
}

#[test]
fn two_servers_sharing_a_name_are_rejected() {
    let e = err("[[mcp.servers]]\nname = \"a\"\nurl = \"http://one\"\n\
                 [[mcp.servers]]\nname = \"a\"\nurl = \"http://two\"\n");
    assert!(e.contains("two servers named"), "{e}");
}

/// The provider request body built from `[model]`, or a failure naming why it was not.
fn params(cfg: &Config) -> serde_json::Value {
    match cfg.model.additional_params() {
        Ok(value) => value,
        Err(e) => unreachable!("expected a request body, got {e}"),
    }
}

#[test]
fn sampling_sends_only_what_is_set() {
    // Everything unset is left to the server's own defaults rather than guessed at here.
    let cfg = ok("[model.sampling]\ntop_p = 0.9\nseed = 7\n");
    let sent = params(&cfg);
    assert_eq!(sent["top_p"], 0.9);
    assert_eq!(sent["seed"], 7);
    assert!(sent.get("top_k").is_none(), "unset fields are not sent");
    assert_eq!(sent["chat_template_kwargs"]["enable_thinking"], false);
}

#[test]
fn extra_params_json_wins_over_sampling() {
    let cfg = ok("[model]\nextra_params_json = \"{\\\"top_p\\\": 0.1}\"\n\
                  [model.sampling]\ntop_p = 0.9\n");
    assert_eq!(params(&cfg)["top_p"], 0.1);
}

#[test]
fn extra_params_that_is_not_a_json_object_is_reported() {
    let cfg = ok("[model]\nextra_params_json = \"[1,2]\"\n");
    match cfg.model.additional_params() {
        Ok(value) => unreachable!("an array is not a request body, got {value}"),
        Err(e) => assert!(e.to_string().contains("JSON object"), "{e}"),
    }
}

/// The system prompt `cfg` resolves to, or a failure naming why it did not resolve.
fn prompt(cfg: &Config) -> String {
    match cfg.system_prompt("built-in") {
        Ok(text) => text,
        Err(e) => unreachable!("expected a prompt, got {e}"),
    }
}

#[test]
fn the_system_prompt_falls_back_then_yields_to_config_then_to_a_file() {
    assert_eq!(prompt(&ok("")), "built-in");
    assert_eq!(prompt(&ok("[prompt]\nsystem = \"  inline  \"\n")), "inline");

    let dir = std::env::temp_dir().join(format!("sparky-prompt-{}", uuid::Uuid::new_v4()));
    let Ok(()) = std::fs::create_dir_all(&dir) else {
        unreachable!("could not create a temp directory")
    };
    let path = dir.join("system.md");
    let Ok(()) = std::fs::write(&path, "from a file\n") else {
        unreachable!("could not write the prompt file")
    };
    let cfg = ok(&format!(
        "[prompt]\nsystem = \"inline\"\nsystem_file = \"{}\"\n",
        path.display()
    ));
    assert_eq!(prompt(&cfg), "from a file", "the file wins over `system`");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_system_prompt_file_fails_at_boot() {
    // Booting with the built-in prompt when the operator asked for a file would answer users
    // in a voice nobody chose.
    let cfg = ok("[prompt]\nsystem_file = \"/nonexistent/sparky-prompt.md\"\n");
    assert!(cfg.system_prompt("built-in").is_err());
}
