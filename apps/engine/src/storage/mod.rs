//! Storage adapters. Postgres is the source of truth; Qdrant is rebuildable.
//! Schema: `apps/engine/migrations`. Imports only `agent::harness`.

pub mod object;
pub mod postgres;
pub mod qdrant;
pub mod redis;
