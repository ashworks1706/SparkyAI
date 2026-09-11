//! Console state and the key map. Rendering is in ui, processes are in runner.

pub mod control;
mod keys;
pub mod ui;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use tokio::sync::mpsc::UnboundedSender;

use crate::core::config::Config;
use crate::core::types::{
    Event, Focus, Health, Kind, LogLine, Mode, ServiceState, Status, Stream, Unit,
};
use crate::units;
use crate::units::logs::{LogBuffer, LogWriter};
use crate::units::runner::Runner;

/// A catalog entry plus what the console knows about it right now.
pub struct UnitState {
    /// The catalog entry.
    pub unit: Unit,
    /// Lifecycle.
    pub status: Status,
    /// Captured output.
    pub logs: LogBuffer,
    /// First visible line when not following.
    pub scroll: usize,
    /// Pin the view to the newest line.
    pub follow: bool,
    /// When it was last started here.
    pub started_at: Option<Instant>,
    /// Start again once the current instance has exited.
    restart_pending: bool,
    /// The console asked it to stop. The exit that follows is not a failure.
    stopping: bool,
}

/// All console state.
pub struct App {
    cfg: Config,
    runner: Runner,
    log_writer: LogWriter,
    /// Units in sidebar order.
    pub units: Vec<UnitState>,
    /// Index into units.
    pub selected: usize,
    /// Input mode.
    pub mode: Mode,
    /// Pane keys act on.
    pub focus: Focus,
    /// The command or search line being typed.
    pub input: String,
    /// Active log search.
    pub search: String,
    /// Line index of the current search hit.
    pub search_hit: Option<usize>,
    /// Latest probes.
    pub health: Health,
    /// Help overlay is open.
    pub help: bool,
    /// One-line notice in the status bar.
    pub notice: Option<String>,
    /// Rows the log pane had at the last draw. Drives paging.
    pub log_rows: usize,
    /// Set by the quit command and the q key.
    pub should_quit: bool,
    /// First half of a two-key chord such as gg.
    pending_key: Option<char>,
}

impl App {
    /// A console over the repo at root.
    pub fn new(cfg: Config, root: PathBuf, tx: &UnboundedSender<Event>) -> anyhow::Result<Self> {
        let log_dir = if cfg.cli.log_dir.is_absolute() {
            cfg.cli.log_dir.clone()
        } else {
            root.join(&cfg.cli.log_dir)
        };
        let log_writer = LogWriter::new(log_dir)?;
        let runner = Runner::new(root, tx.clone());
        let units = units::catalog()
            .into_iter()
            .map(|u| UnitState::new(u, cfg.cli.log_lines))
            .collect();
        Ok(Self {
            cfg,
            runner,
            log_writer,
            units,
            selected: 0,
            mode: Mode::Normal,
            focus: Focus::Units,
            input: String::new(),
            search: String::new(),
            search_hit: None,
            health: Health::default(),
            help: false,
            notice: None,
            log_rows: 20,
            should_quit: false,
            pending_key: None,
        })
    }

    /// Phoenix UI location, for the status bar.
    pub fn phoenix_url(&self) -> &str {
        &self.cfg.cli.phoenix_url
    }

    /// Engine location, for the status bar.
    pub fn engine_url(&self) -> &str {
        &self.cfg.engine.base_url
    }

    /// The selected unit.
    pub fn current(&self) -> &UnitState {
        &self.units[self.selected.min(self.units.len() - 1)]
    }

    fn current_mut(&mut self) -> &mut UnitState {
        let i = self.selected.min(self.units.len() - 1);
        &mut self.units[i]
    }

    /// Kills host processes before the terminal is restored.
    pub fn shutdown(&mut self) {
        self.runner.shutdown();
    }

    /// Applies one event.
    pub fn handle(&mut self, event: Event) {
        match event {
            Event::Key(key) => self.key(key),
            Event::Tick | Event::Resize => {}
            Event::Log { unit, line } => self.log(&unit, line),
            Event::Exited { unit, code } => self.exited(&unit, code),
            Event::Services(Ok(states)) => self.services(&states),
            Event::Services(Err(e)) => self.notice = Some(e),
            Event::Health(h) => self.health = h,
            Event::InputLost(why) => {
                self.notice = Some(format!("terminal input ended ({why}); quitting"));
                self.should_quit = true;
            }
        }
    }

    fn log(&mut self, unit: &str, line: LogLine) {
        if let Err(e) = self.log_writer.append(unit, &line) {
            self.notice = Some(format!("write log: {e}"));
        }
        if let Some(u) = self.units.iter_mut().find(|u| u.unit.id == unit) {
            u.logs.push(line);
        }
    }

    fn exited(&mut self, unit: &str, code: Option<i32>) {
        self.runner.forget(unit);
        let (note, restart_id) = {
            let Some(u) = self.units.iter_mut().find(|u| u.unit.id == unit) else {
                return;
            };
            let note = match u.unit.kind {
                Kind::Service { .. } => {
                    if let Some(c) = code.filter(|c| *c != 0) {
                        u.status = Status::Failed(format!("compose exited {c}"));
                    }
                    None
                }
                Kind::Process | Kind::Task => {
                    let stopped = std::mem::take(&mut u.stopping);
                    u.status = match code {
                        Some(c) if !stopped => Status::Exited(c),
                        _ => Status::Stopped,
                    };
                    Some(match code {
                        Some(c) if !stopped => format!("exited with {c}"),
                        _ => "stopped".to_owned(),
                    })
                }
            };
            let restart_id = std::mem::take(&mut u.restart_pending).then(|| u.unit.id.clone());
            (note, restart_id)
        };
        if let Some(note) = note {
            self.log(unit, LogLine::now(Stream::Meta, note));
        }
        if let Some(id) = restart_id {
            self.start_by_id(&id);
        }
    }

    fn services(&mut self, states: &HashMap<String, ServiceState>) {
        for u in &mut self.units {
            let Some(service) = u.unit.service() else {
                continue;
            };
            match states.get(service) {
                Some(s) => u.status = s.status(),
                None if u.status == Status::Starting => {}
                None => u.status = Status::Stopped,
            }
        }
    }

    fn index_of(&self, id: &str) -> Option<usize> {
        self.units.iter().position(|u| u.unit.id == id)
    }
}

impl UnitState {
    fn new(unit: Unit, log_lines: usize) -> Self {
        Self {
            unit,
            status: Status::Stopped,
            logs: LogBuffer::new(log_lines),
            scroll: 0,
            follow: true,
            started_at: None,
            restart_pending: false,
            stopping: false,
        }
    }
}
