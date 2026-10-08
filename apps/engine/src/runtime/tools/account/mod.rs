//! Per-user integrations: tools over the caller's own accounts, and the OAuth grants they read.

pub mod canvas;
pub mod gcal;
pub mod grant;
#[allow(
    dead_code,
    reason = "called by the per-user session routes of roadmap phase 8"
)]
pub mod oauth;
pub mod outlook;
