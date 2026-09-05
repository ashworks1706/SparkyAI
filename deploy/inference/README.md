# deploy/inference

Model serving: `llama-server` from llama.cpp, configured here. Every model speaks the
OpenAI-compatible API, the only surface `apps/engine` and `apps/scraper` use.

`llama-server` serves one model per process, so each role is its own container.

| Role | Default GGUF | Port | Flags |
|---|---|---|---|
| chat | `Qwen/Qwen3-4B-GGUF:Q4_K_M` | 8000 | `--jinja` for tool calls, `--metrics`, `--ctx-size 8192`, `--parallel 2` |
| embeddings | `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` | 8001 | `--embeddings --pooling last`, batch 1024, `--parallel 2` |

Override either with `SPARKY_CHAT_GGUF` or `SPARKY_EMBED_GGUF`.

## Concurrency

`--parallel N` gives the server N slots. Continuous batching interleaves their decode steps in
one forward pass, so N requests generate at once over one copy of the weights.

`--ctx-size` is the total across slots. Per-slot context is `ctx-size / parallel`, and the KV
cache scales with the total:

```
SPARKY_CHAT_PARALLEL=4 SPARKY_CHAT_CTX=16384   # 4 slots x 4096 tokens
```

Aggregate throughput rises sublinearly with N; per-request generation gets slower as slots fill.

`SPARKY_AGENT__MODEL_SLOTS` in the engine holds the same number. The engine admits that many
model calls at once and queues the rest for `SPARKY_AGENT__MODEL_QUEUE_WAIT_SECS`, then answers
503. Set it to 0 to remove the engine-side limit and let `llama-server` queue.

## Local

```bash
just model     # starts both; GGUFs download on first run into the modelcache volume
```

Endpoints: `http://localhost:8000/v1` (chat) and `:8001/v1` (embeddings). From inside compose
the hosts are `chat` and `embed`.

On a 6 GB card both fit with room to spare: embed takes ~0.9 GB, chat ~3.1 GB (2.5 GB of
weights plus a KV cache of ~576 MB per 4096 tokens). At the default `--ctx-size 8192
--parallel 2` the chat KV cache is ~1.2 GB. Raise `SPARKY_CHAT_CTX` only with headroom, and
lower `--n-gpu-layers` to spill to CPU.

## Deployed

Same two services on a GPU host with a larger chat GGUF. Size the GPU for the weights plus a
KV cache proportional to `--ctx-size`. Rough starting points:

| VRAM | `SPARKY_CHAT_PARALLEL` | `SPARKY_CHAT_CTX` |
|---|---|---|
| 6-8 GB | 2 | 8192 |
| 24 GB | 4 | 16384 |
| 40-80 GB | 8 | 32768 |

Raise `SPARKY_AGENT__MODEL_SLOTS` to match. Past one card's slots, run more `llama-server`
replicas behind a load balancer; each replica holds its own copy of the weights.
