//! Per-user integrations: tools over the caller's own accounts, and the OAuth grants they read.

#[cfg(feature = "standalone")]
pub mod canvas;
pub mod gcal;
pub mod grant;
pub mod oauth;
pub mod outlook;
