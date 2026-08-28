//! Storage adapters. Postgres is the source of truth; Qdrant is rebuildable. See `docs/ARCHITECTURE.md`.

pub mod object;
pub mod postgres;
pub mod qdrant;
pub mod redis;
