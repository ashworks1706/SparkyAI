//! `harness::model::ModelProvider`, `Embedder`, and `Reranker` implementations.
//! Chat and embeddings go through Rig's OpenAI-compatible client; rerank is a direct HTTP
//! call because Rig has no provider for llama-server's `/v1/rerank`.

pub mod rerank;
pub mod rig_openai;
