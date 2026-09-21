//! Console state and the key map. Rendering is in ui, processes are in runner.

pub mod control;
mod keys;
pub mod ui;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use tokio::sync::mpsc::UnboundedSender;

use crate::core::config::Config;
use crate::core::types::{
    Event, Focus, Health, Kind, LogLine, Mode, SANDBOX_SWITCH, SandboxCommand, SandboxReport,
    SandboxUnit, ServiceState, Status, Stream, Unit,
};
use crate::units;
use crate::units::logs::{LogBuffer, LogWriter};
use crate::units::runner::Runner;
use crate::units::sandbox;

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
    /// Sends events back from the tasks the console starts.
    pub(super) tx: UnboundedSender<Event>,
    /// Where the engine sandbox routes are.
    pub(super) sandbox: sandbox::Endpoint,
    /// Command id to whether its end was shown, so each one is written at most twice.
    shown_commands: HashMap<u64, bool>,
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
        let log_writer = LogWriter::new(log_dir, cfg.cli.log_file_max_mb * 1024 * 1024)?;
        let runner = Runner::new(root, tx.clone(), cfg.cli.log_line_chars);
        let sandbox = sandbox::Endpoint::from_config(&cfg);
        let units = units::catalog()
            .into_iter()
            .map(|u| UnitState::new(u, cfg.cli.log_lines))
            .collect();
        Ok(Self {
            cfg,
            runner,
            tx: tx.clone(),
            sandbox,
            shown_commands: HashMap::new(),
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

    /// Phoenix trace UI location, for the status bar.
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
            Event::Sandbox(report) => self.sandbox_report(report),
            Event::SandboxActed(Ok(note)) => self.notice = Some(note),
            Event::SandboxActed(Err(why)) => self.notice = Some(why),
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
                // The engine owns these; a report says what they are doing.
                Kind::Sandbox(_) => None,
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

    /// Brings the sandbox rows in line with what the engine reports, and shows new commands.
    fn sandbox_report(&mut self, report: Result<SandboxReport, String>) {
        let report = match report {
            Ok(report) => report,
            Err(why) => {
                // The engine being down is the usual reason; the status bar already says so.
                self.drop_sandbox_rows(&[]);
                self.upsert_unit(units::sandbox_switch(), Status::Failed(why));
                return;
            }
        };
        self.upsert_unit(units::sandbox_switch(), Status::from_on(report.enabled));
        for session in &report.sessions {
            self.upsert_unit(units::sandbox_session(session), Status::Running);
        }
        let live: Vec<String> = report.sessions.iter().map(|s| s.name.clone()).collect();
        self.drop_sandbox_rows(&live);
        self.show_commands(&report.commands);
    }

    /// Writes the commands newer than the newest already shown into the switch log, oldest first.
    fn show_commands(&mut self, commands: &[SandboxCommand]) {
        for command in sandbox::unwritten(commands, &self.shown_commands) {
            self.shown_commands.insert(command.id, command.ended());
            let line = LogLine::now(Stream::Out, command.line());
            self.log(SANDBOX_SWITCH, line.clone());
            let Some(container) = &command.container else {
                continue;
            };
            if let Some(u) = self.units.iter_mut().find(|u| match &u.unit.kind {
                Kind::Sandbox(SandboxUnit::Session { container: held }) => held == container,
                _ => false,
            }) {
                u.logs.push(line);
            }
        }
        // A command that has scrolled out of the engine log is forgotten here too.
        let live: HashSet<u64> = commands.iter().map(|c| c.id).collect();
        self.shown_commands.retain(|id, _| live.contains(id));
    }

    /// Adds a unit, or refreshes the one already listed under its id.
    fn upsert_unit(&mut self, unit: Unit, status: Status) {
        if let Some(i) = self.index_of(&unit.id) {
            self.units[i].unit = unit;
            self.units[i].status = status;
            return;
        }
        let mut state = UnitState::new(unit, self.cfg.cli.log_lines);
        state.status = status;
        self.units.push(state);
    }

    /// Takes back the rows of session containers the engine no longer reports.
    fn drop_sandbox_rows(&mut self, live: &[String]) {
        let was = self.selected;
        let selected = self.units.get(was).map(|u| u.unit.id.clone());
        self.units.retain(|u| match &u.unit.kind {
            Kind::Sandbox(SandboxUnit::Session { container }) => live.contains(container),
            _ => true,
        });
        let Some(last) = self.units.len().checked_sub(1) else {
            self.selected = 0;
            return;
        };
        // The cursor follows its row, and stays where it was when that row is the one that went.
        self.selected = selected
            .and_then(|id| self.index_of(&id))
            .unwrap_or(was)
            .min(last);
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
