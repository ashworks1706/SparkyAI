//! Discord bot: slash commands → HTTP calls to `sparky api` → streamed reply with citations.
//! A client of the API; never links the harness.

pub mod api_client;
pub mod bot;
pub mod commands;
pub mod reply;
