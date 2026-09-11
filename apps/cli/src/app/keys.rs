//! Key handling and the navigation it drives.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::App;
use super::control::parse_command;
use crate::core::types::{Command, Focus, Mode, Status};

impl App {
    pub(super) fn key(&mut self, key: KeyEvent) {
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
    pub(super) fn key_normal(&mut self, key: KeyEvent) {
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
    pub(super) fn key_line(&mut self, key: KeyEvent, command: bool) {
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
    pub(super) fn down(&mut self, n: usize) {
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
    pub(super) fn up(&mut self, n: usize) {
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
    pub(super) fn top(&mut self) {
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
    pub(super) fn bottom(&mut self) {
        match self.focus {
            Focus::Units => {
                self.selected = self.units.len() - 1;
                self.on_select();
            }
            Focus::Logs => self.current_mut().follow = true,
        }
    }
    pub(super) fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Units => Focus::Logs,
            Focus::Logs => Focus::Units,
        };
    }
    pub(super) fn search_step(&mut self, backwards: bool) {
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
    pub(super) fn open_url(&mut self) {
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
    /// Follows the container logs of a running service when it is selected.
    pub(super) fn on_select(&mut self) {
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
}
