//! The catalog: every unit the repo can run, in sidebar order.

use crate::core::types::{Group, Kind, Unit};

fn service(id: &str, profile: Option<&str>, hint: &str, url: Option<&str>) -> Unit {
    Unit {
        id: id.into(),
        group: match profile {
            Some("model") => Group::Models,
            Some(_) => Group::Tools,
            None => Group::Infra,
        },
        kind: Kind::Service {
            service: id.into(),
            profile: profile.map(Into::into),
        },
        args: Vec::new(),
        hint: hint.into(),
        url: url.map(Into::into),
    }
}

fn process(id: &str, args: &[&str], hint: &str, url: Option<&str>) -> Unit {
    Unit {
        id: id.into(),
        group: Group::Apps,
        kind: Kind::Process,
        args: args.iter().map(|s| (*s).into()).collect(),
        hint: hint.into(),
        url: url.map(Into::into),
    }
}

/// A one-shot recipe; the id is the recipe line itself.
pub fn task(args: &[String], hint: &str) -> Unit {
    Unit {
        id: args.join(" "),
        group: Group::Tasks,
        kind: Kind::Task,
        args: args.to_vec(),
        hint: hint.into(),
        url: None,
    }
}

fn task_static(args: &[&str], hint: &str) -> Unit {
    let args: Vec<String> = args.iter().map(|s| (*s).into()).collect();
    task(&args, hint)
}

/// A recipe in the deploy section; `follows` marks the ones that stream until stopped.
fn deploy(args: &[&str], hint: &str, follows: bool) -> Unit {
    let mut unit = task_static(args, hint);
    unit.group = Group::Deploy;
    if follows {
        unit.kind = Kind::Process;
    }
    unit
}

/// Everything the console knows how to run.
pub fn catalog() -> Vec<Unit> {
    vec![
        service(
            "postgres",
            None,
            "pgvector 17; the retrieval index and conversations",
            Some("localhost:5432"),
        ),
        service("redis", None, "queue backend", Some("localhost:6379")),
        service("minio", None, "object store", Some("http://localhost:9001")),
        service(
            "phoenix",
            None,
            "traces from every app",
            Some("http://localhost:6006"),
        ),
        service(
            "chat",
            Some("model"),
            "llama-server chat model",
            Some("http://localhost:8000"),
        ),
        service(
            "embed",
            Some("model"),
            "llama-server embedding model",
            Some("http://localhost:8001"),
        ),
        service(
            "firecrawl",
            Some("crawl"),
            "self-hosted Firecrawl for the scraper",
            Some("http://localhost:3002"),
        ),
        service(
            "playwright-mcp",
            Some("browser"),
            "browser tools for the engine",
            Some("http://127.0.0.1:8931"),
        ),
        process(
            "engine",
            &["engine"],
            "the agent and its HTTP surface",
            Some("http://localhost:8080"),
        ),
        process(
            "discord",
            &["discord"],
            "serenity bot, HTTP client of the engine",
            None,
        ),
        process(
            "web",
            &["web"],
            "Vite dev server",
            Some("http://localhost:5173"),
        ),
        task_static(&["doctor"], "required tools, .env, hooks"),
        task_static(&["setup"], "install every unit's deps"),
        task_static(&["migrate"], "apply scraper migrations"),
        task_static(
            &["scraper", "run", "library_hours"],
            "fetch, chunk, embed one source",
        ),
        task_static(&["check"], "fmt, lint, test every unit"),
        task_static(
            &["data", "export"],
            "Phoenix llm spans → .sparky/training/data",
        ),
        task_static(&["data", "verify"], "verify and deduplicate training data"),
        task_static(&["data", "stats"], "dataset summary"),
        task_static(&["eval", "run"], "golden cases against the engine"),
        task_static(&["eval", "baseline"], "promote the last report"),
        task_static(&["eval", "compare"], "fail on regression"),
        task_static(&["train", "sft", "--dry-run"], "validate config and data"),
        task_static(&["train", "sft"], "QLoRA → GGUF (GPU)"),
        deploy(
            &["up"],
            "dev stack: build images locally, start everything",
            false,
        ),
        deploy(&["down"], "stop the whole stack, every profile", false),
        deploy(&["ps"], "what compose has, across every profile", false),
        deploy(&["logs"], "follow every container", true),
        deploy(&["images"], "build the rust and scraper images", false),
        deploy(
            &["prod-up"],
            "RunPod host: pull GHCR images (SPARKY_IMAGE_TAG) and start",
            false,
        ),
        deploy(&["prod-down"], "RunPod host: stop the prod stack", false),
        deploy(&["prod-logs"], "RunPod host: follow the prod stack", true),
    ]
}
