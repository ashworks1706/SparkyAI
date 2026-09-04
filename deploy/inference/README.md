# deploy/inference

Model serving. Not our code — Ollama — but a deployable we own the configuration for.

The same Ollama server backs local development and deployment; only the host and the model
sizes differ. It speaks the OpenAI-compatible API on `:11434/v1`, which is the only surface
`apps/engine` and `apps/knowledge` know about.

| Role | Model | Env var | Endpoint |
|---|---|---|---|
| chat | `qwen3:4b` local, `qwen3:14b` deployed | `SPARKY_CHAT_MODEL` | `/v1/chat/completions` |
| embeddings | `qwen3-embedding:0.6b` (1024-dim) | `SPARKY_EMBED_MODEL` | `/v1/embeddings` |

## Local

```bash
just model     # starts the ollama container and pulls both models
```

Point `SPARKY_MODEL__BASE_URL` and `SPARKY_EMBEDDING__BASE_URL` at `http://localhost:11434/v1`
(`http://ollama:11434/v1` from inside compose).

## Deployed

Run the same `ollama/ollama` image on a GPU host, pull the larger chat model, and point the
same variables at it. `OLLAMA_KEEP_ALIVE` controls how long a model stays resident; set it to
`-1` in deployment so the first request after an idle period is not paying a load.

Ollama serializes concurrent generations per model. Under real load, run more replicas or set
`OLLAMA_NUM_PARALLEL` and size the GPU for the resulting KV cache.

## Reranking

Ollama has no rerank endpoint. Phase 2 needs one; the options are a `llama.cpp` server started
with `--reranking`, or a cross-encoder loaded in-process by `apps/knowledge`. Open decision.
