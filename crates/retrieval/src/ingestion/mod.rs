//! Offline jobs: fetch → snapshot → extract → normalize → dedupe → chunk → embed → index.

pub mod chunk;
pub mod extract;
pub mod fetch;
pub mod index;
pub mod sources;
