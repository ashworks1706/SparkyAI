//! Starting, stopping and restarting a unit, and the commands that ask for it.

use std::time::{Duration, Instant};

use super::{App, UnitState};
use crate::core::types::{Command, Focus, Kind, Status};
use crate::units;
use crate::units::health;

impl App {
    pub(super) fn toggle_selected(&mut self) {
        if self.current().status.is_active() {
            self.stop_selected();
        } else {
            let id = self.current().unit.id.clone();
            self.start_by_id(&id);
        }
    }
    pub(super) fn stop_selected(&mut self) {
        let id = self.current().unit.id.clone();
        self.stop_by_id(&id);
    }
    pub(super) fn restart_selected(&mut self) {
        let id = self.current().unit.id.clone();
        self.run(Command::Restart(id));
    }
    pub(super) fn start_by_id(&mut self, id: &str) {
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
        if let Some(addr) = self.taken_port(&unit) {
            self.notice = Some(format!(
                "{id} is already served on {addr} by something this console did not start"
            ));
            return;
        }
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
    /// The address a process unit would bind, when something already answers there. A compose
    /// service is left alone: bringing one up that is already up changes nothing.
    fn taken_port(&self, unit: &crate::core::types::Unit) -> Option<String> {
        if !matches!(unit.kind, Kind::Process) || self.runner.owns(&unit.id) {
            return None;
        }
        let addr = health::address_of(unit.url.as_deref()?)?;
        let timeout = Duration::from_millis(self.cfg.cli.port_check_ms.max(1));
        health::served(&addr, timeout).then_some(addr)
    }

    pub(super) fn stop_by_id(&mut self, id: &str) {
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
    pub(super) fn run(&mut self, cmd: Command) {
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
    pub(super) fn run_adhoc(&mut self, args: &[String]) {
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

/// Parses the text typed after a colon.
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
