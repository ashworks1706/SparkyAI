//! Storage adapters. Postgres is the source of truth; Qdrant is rebuildable.
//! Schema: `apps/api/migrations`.

pub mod object;
pub mod postgres;
pub mod qdrant;
pub mod redis;
