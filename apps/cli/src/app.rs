//! Console state and the key map. Rendering is in `ui`; processes are in `runner`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::sync::mpsc::UnboundedSender;

use crate::core::config::Config;
use crate::core::types::{
    Command, Event, Focus, Health, Kind, LogLine, Mode, ServiceState, Status, Stream, Unit,
};
use crate::logs::{LogBuffer, LogWriter};
use crate::runner::Runner;
use crate::units;

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
    /// The console asked it to stop; the exit that follows is not a failure.
    stopping: bool,
}

/// All console state.
pub struct App {
    cfg: Config,
    runner: Runner,
    log_writer: LogWriter,
    /// Units in sidebar order.
    pub units: Vec<UnitState>,
    /// Index into `units`.
    pub selected: usize,
    /// Input mode.
    pub mode: Mode,
    /// Pane keys act on.
    pub focus: Focus,
    /// The `:` or `/` line being typed.
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
    /// Rows the log pane had at the last draw; drives paging.
    pub log_rows: usize,
    /// Set by `:q` and `q`.
    pub should_quit: bool,
    /// First half of a two-key chord such as `gg`.
    pending_key: Option<char>,
}

impl App {
    /// A console over the repo at `root`.
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

    fn key(&mut self, key: KeyEvent) {
        self.notice = None;
        if self.help {
            self.help = false;
            return;
        }
        match self.mode {
            Mode::Normal => self.key_normal(key),
            Mode::Command => self.key_line(key, true),
            Mode::Search => self.key_line(key, false),
        }
    }

    fn key_normal(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let g = self.pending_key.take() == Some('g');
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('?') => self.help = true,
            KeyCode::Esc => self.notice = None,
            KeyCode::Char('j') | KeyCode::Down => self.down(1),
            KeyCode::Char('k') | KeyCode::Up => self.up(1),
            KeyCode::Char('d') if ctrl => self.down(self.log_rows / 2),
            KeyCode::Char('u') if ctrl => self.up(self.log_rows / 2),
            KeyCode::Char('g') if g => self.top(),
            KeyCode::Char('g') => self.pending_key = Some('g'),
            KeyCode::Char('G') => self.bottom(),
            KeyCode::Enter | KeyCode::Char('s') => self.toggle_selected(),
            KeyCode::Char('x') => self.stop_selected(),
            KeyCode::Char('r') => self.restart_selected(),
            KeyCode::Char('h') | KeyCode::Left => self.focus = Focus::Units,
            KeyCode::Char('l') | KeyCode::Right => self.focus = Focus::Logs,
            KeyCode::Tab => self.cycle_focus(),
            KeyCode::Char('C') => self.run(Command::Clear),
            KeyCode::Char('o') => self.open_url(),
            KeyCode::Char(':') => {
                self.mode = Mode::Command;
                self.input.clear();
            }
            KeyCode::Char('/') => {
                self.mode = Mode::Search;
                self.focus = Focus::Logs;
                self.input.clear();
            }
            KeyCode::Char('n') => self.search_step(false),
            KeyCode::Char('N') => self.search_step(true),
            _ => {}
        }
    }

    fn key_line(&mut self, key: KeyEvent, command: bool) {
        match key.code {
            KeyCode::Esc => self.mode = Mode::Normal,
            KeyCode::Enter => {
                let text = std::mem::take(&mut self.input);
                self.mode = Mode::Normal;
                if command {
                    self.run(parse_command(&text));
                } else {
                    self.search = text;
                    self.search_hit = None;
                    self.search_step(false);
                }
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.clear();
            }
            KeyCode::Char(c) => self.input.push(c),
            _ => {}
        }
    }

    fn down(&mut self, n: usize) {
        match self.focus {
            Focus::Units => {
                self.selected = (self.selected + n).min(self.units.len() - 1);
                self.on_select();
            }
            Focus::Logs => {
                let rows = self.log_rows;
                let u = self.current_mut();
                let max_top = u.logs.len().saturating_sub(rows);
                u.scroll = (u.scroll + n).min(max_top);
                u.follow = u.scroll >= max_top;
            }
        }
    }

    fn up(&mut self, n: usize) {
        match self.focus {
            Focus::Units => {
                self.selected = self.selected.saturating_sub(n);
                self.on_select();
            }
            Focus::Logs => {
                let rows = self.log_rows;
                let u = self.current_mut();
                if u.follow {
                    u.scroll = u.logs.len().saturating_sub(rows);
                }
                u.scroll = u.scroll.saturating_sub(n);
                u.follow = false;
            }
        }
    }

    fn top(&mut self) {
        match self.focus {
            Focus::Units => {
                self.selected = 0;
                self.on_select();
            }
            Focus::Logs => {
                let u = self.current_mut();
                u.scroll = 0;
                u.follow = false;
            }
        }
    }

    fn bottom(&mut self) {
        match self.focus {
            Focus::Units => {
                self.selected = self.units.len() - 1;
                self.on_select();
            }
            Focus::Logs => self.current_mut().follow = true,
        }
    }

    fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Units => Focus::Logs,
            Focus::Logs => Focus::Units,
        };
    }

    /// Follows a running service's container logs when it is selected.
    fn on_select(&mut self) {
        self.search_hit = None;
        let u = self.current();
        if u.status == Status::Running
            && let Some(service) = u.unit.service()
        {
            let service = service.to_owned();
            if let Err(e) = self.runner.follow(&service) {
                self.notice = Some(e.to_string());
            }
        }
    }

    fn open_url(&mut self) {
        let Some(url) = self.current().unit.url.clone() else {
            self.notice = Some("no URL for this unit".into());
            return;
        };
        let url = if url.starts_with("http") {
            url
        } else {
            format!("http://{url}")
        };
        match std::process::Command::new("xdg-open")
            .arg(&url)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(_) => self.notice = Some(format!("opened {url}")),
            Err(e) => self.notice = Some(format!("xdg-open: {e}")),
        }
    }

    fn search_step(&mut self, backwards: bool) {
        if self.search.is_empty() {
            self.notice = Some("no search; press / first".into());
            return;
        }
        let rows = self.log_rows;
        let needle = self.search.clone();
        let from = self.search_hit;
        let u = self.current_mut();
        let start = from.unwrap_or_else(|| {
            if backwards {
                0
            } else {
                u.logs.len().saturating_sub(1)
            }
        });
        match u.logs.find(&needle, start, backwards) {
            Some(i) => {
                u.follow = false;
                u.scroll = i.saturating_sub(rows / 2);
                self.search_hit = Some(i);
                self.focus = Focus::Logs;
            }
            None => self.notice = Some(format!("no match for {needle:?}")),
        }
    }

    fn toggle_selected(&mut self) {
        if self.current().status.is_active() {
            self.stop_selected();
        } else {
            let id = self.current().unit.id.clone();
            self.start_by_id(&id);
        }
    }

    fn stop_selected(&mut self) {
        let id = self.current().unit.id.clone();
        self.stop_by_id(&id);
    }

    fn restart_selected(&mut self) {
        let id = self.current().unit.id.clone();
        self.run(Command::Restart(id));
    }

    fn index_of(&self, id: &str) -> Option<usize> {
        self.units.iter().position(|u| u.unit.id == id)
    }

    fn start_by_id(&mut self, id: &str) {
        let Some(i) = self.index_of(id) else {
            self.notice = Some(format!("no unit {id:?}"));
            return;
        };
        if self.units[i].status.is_active() {
            self.notice = Some(format!("{id} is already running"));
            return;
        }
        let unit = self.units[i].unit.clone();
        self.selected = i;
        match self.runner.start(&unit) {
            Ok(()) => {
                let u = &mut self.units[i];
                u.status = match unit.kind {
                    Kind::Service { .. } => Status::Starting,
                    Kind::Process | Kind::Task => Status::Running,
                };
                u.started_at = Some(Instant::now());
                u.follow = true;
                self.notice = Some(format!("started {id}"));
            }
            Err(e) => {
                self.units[i].status = Status::Failed(e.to_string());
                self.notice = Some(e.to_string());
            }
        }
    }

    fn stop_by_id(&mut self, id: &str) {
        let Some(i) = self.index_of(id) else {
            self.notice = Some(format!("no unit {id:?}"));
            return;
        };
        let unit = self.units[i].unit.clone();
        if !self.units[i].status.is_active() && !self.runner.owns(id) {
            self.notice = Some(format!("{id} is not running"));
            return;
        }
        match self.runner.stop(&unit) {
            Ok(()) => {
                self.units[i].stopping = true;
                self.notice = Some(format!("stopping {id}"));
            }
            Err(e) => self.notice = Some(e.to_string()),
        }
    }

    /// Executes a parsed command.
    pub fn run(&mut self, cmd: Command) {
        match cmd {
            Command::Quit => self.should_quit = true,
            Command::Start(id) => self.start_by_id(&id),
            Command::Stop(id) => self.stop_by_id(&id),
            Command::Restart(id) => {
                if let Some(i) = self.index_of(&id) {
                    if self.units[i].status.is_active() || self.runner.owns(&id) {
                        self.units[i].restart_pending = true;
                        self.stop_by_id(&id);
                    } else {
                        self.start_by_id(&id);
                    }
                } else {
                    self.notice = Some(format!("no unit {id:?}"));
                }
            }
            Command::Just(args) => self.run_adhoc(&args),
            Command::Help => self.help = true,
            Command::Clear => self.current_mut().logs.clear(),
            Command::Unknown(text) => {
                self.notice = Some(format!("unknown command {text:?}; try :help"));
            }
        }
    }

    fn run_adhoc(&mut self, args: &[String]) {
        if args.is_empty() {
            self.notice = Some("usage: :just <recipe> [args]".into());
            return;
        }
        let unit = units::task(args, "ad-hoc");
        let i = if let Some(i) = self.index_of(&unit.id) {
            i
        } else {
            self.units
                .push(UnitState::new(unit.clone(), self.cfg.cli.log_lines));
            self.units.len() - 1
        };
        self.selected = i;
        self.focus = Focus::Logs;
        self.start_by_id(&unit.id);
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

/// Parses the text typed after `:`.
pub fn parse_command(text: &str) -> Command {
    let mut words = text.split_whitespace();
    let Some(head) = words.next() else {
        return Command::Unknown(String::new());
    };
    let rest: Vec<String> = words.map(str::to_owned).collect();
    let arg = rest.join(" ");
    match head {
        "q" | "quit" => Command::Quit,
        "start" | "up" if !arg.is_empty() => Command::Start(arg),
        "stop" | "down" if !arg.is_empty() => Command::Stop(arg),
        "restart" if !arg.is_empty() => Command::Restart(arg),
        "just" => Command::Just(rest),
        "help" | "h" => Command::Help,
        "clear" => Command::Clear,
        _ => Command::Just(std::iter::once(head.to_owned()).chain(rest).collect()),
    }
}
