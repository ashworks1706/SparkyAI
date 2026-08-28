# 0002 — Model, embedding, reranker, and observability choices

**Chat model.** `Qwen/Qwen3-14B` on vLLM with `--enable-auto-tool-choice --tool-call-parser hermes`. 14B fits one A100 80GB / H100 at 32k context. Swap by changing `SPARKY_MODEL__NAME`; nothing else references the model.

**Embeddings.** `Qwen/Qwen3-Embedding-0.6B`, 1024-dim, served by vLLM `--task embed`. Same family as the chat model, small enough for an L4. Dimension is config (`SPARKY_EMBEDDING__DIM`); changing model requires a Qdrant collection rebuild, which `source_versions.embedding_model` makes traceable.

**Reranker.** `Qwen/Qwen3-Reranker-0.6B`, vLLM `--task score`, colocated with embeddings.

**Errors.** Sentry via `sentry` + `sentry-tracing`; `tracing::error!` events become Sentry events. Optional — unset DSN disables it.

**Traces.** OpenTelemetry over OTLP/gRPC to Axiom (`x-axiom-dataset` header). Any OTLP backend works by changing the endpoint and dropping the Axiom headers. Optional.

**Logs.** `tracing-subscriber`: pretty in development, JSON elsewhere; shipped by the platform, not the app.

**Object storage.** MinIO locally; any S3-compatible bucket in deployment.

**Postgres.** 17, via `sqlx` with migrations in `apps/engine/migrations`, applied by the engine at startup.
