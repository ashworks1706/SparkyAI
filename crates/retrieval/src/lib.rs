//! Query-side retrieval: embed → dense + BM25 → fuse → rerank → `Evidence`.
//! Ingestion lives in `services/ingest`. See `docs/ARCHITECTURE.md`.

pub mod embed;
pub mod hybrid;
pub mod rerank;
