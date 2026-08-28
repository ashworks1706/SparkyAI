//! The thing that finds: embed → dense (Qdrant) + BM25 → fuse → rerank → `Evidence`.
//! Ingestion lives in `apps/ingest`. Imports only `agent::harness`.

pub mod embed;
pub mod hybrid;
pub mod rerank;
