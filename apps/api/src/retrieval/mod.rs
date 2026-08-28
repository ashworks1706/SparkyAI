//! Query-side retrieval: embed → dense + BM25 → fuse → rerank → `Evidence`.
//! Ingestion lives in `apps/ingest`.

pub mod embed;
pub mod hybrid;
pub mod rerank;
