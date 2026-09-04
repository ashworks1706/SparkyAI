//! `sparky`: the developer console. Runs every unit in the repo, streams its logs, probes the
//! engine and its dependencies, and chats with the agent, from one modal terminal UI.

mod app;
mod chat;
mod core;
mod health;
mod logs;
mod runner;
mod ui;
mod units;

use std::time::Duration;

use crossterm::event::{Event as TermEvent, EventStream, KeyEventKind};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::app::App;
use crate::core::types::Event;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cwd = std::env::current_dir()?;
    let root = core::config::repo_root(&cwd).ok_or_else(|| {
        anyhow::anyhow!(
            "run from inside the SparkyAI repo (no justfile found above {})",
            cwd.display()
        )
    })?;
    let _ = dotenvy::from_path(root.join(".env"));
    let cfg = core::config::load()?;
    let targets = health::Targets::from_config(&cfg);

    let (tx, mut rx) = mpsc::unbounded_channel::<Event>();
    let mut app = App::new(cfg, root.clone(), tx.clone())?;

    tokio::spawn(health::poll(targets, tx.clone()));
    tokio::spawn(services_task(root, tx.clone()));
    tokio::spawn(ticker(tx.clone()));
    tokio::spawn(keys(tx));

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        original_hook(info);
    }));
    let mut terminal = ratatui::init();
    let outcome = run(&mut terminal, &mut app, &mut rx).await;
    app.shutdown();
    ratatui::restore();
    outcome
}

async fn run(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<Event>,
) -> anyhow::Result<()> {
    terminal.draw(|f| ui::draw(f, app))?;
    while let Some(event) = rx.recv().await {
        app.handle(event);
        // Drain whatever else is queued so a burst of log lines costs one redraw.
        while let Ok(more) = rx.try_recv() {
            app.handle(more);
        }
        if app.should_quit {
            break;
        }
        terminal.draw(|f| ui::draw(f, app))?;
    }
    Ok(())
}

async fn ticker(tx: mpsc::UnboundedSender<Event>) {
    let mut interval = tokio::time::interval(Duration::from_millis(500));
    loop {
        interval.tick().await;
        if tx.send(Event::Tick).is_err() {
            return;
        }
    }
}

async fn keys(tx: mpsc::UnboundedSender<Event>) {
    let mut stream = EventStream::new();
    while let Some(Ok(event)) = stream.next().await {
        let mapped = match event {
            TermEvent::Key(k) if k.kind == KeyEventKind::Press => Event::Key(k),
            TermEvent::Resize(_, _) => Event::Resize,
            _ => continue,
        };
        if tx.send(mapped).is_err() {
            return;
        }
    }
}

async fn services_task(root: std::path::PathBuf, tx: mpsc::UnboundedSender<Event>) {
    loop {
        let states = runner::Runner::service_states(&root).await;
        if tx.send(Event::Services(states)).is_err() {
            return;
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}
