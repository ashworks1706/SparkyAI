# deploy/inference

Model serving. Not our code — `llama-server` from llama.cpp — but a deployable we own the
configuration for. Every model speaks the OpenAI-compatible API, which is the only surface
`apps/engine` and `apps/knowledge` know about.

`llama-server` serves one model per process, so each role is its own container.

| Role | Default GGUF | Port | Flags |
|---|---|---|---|
| chat | `Qwen/Qwen3-4B-GGUF:Q4_K_M` | 8000 | `--jinja` for tool calls, `--ctx-size 4096` |
| embeddings | `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` | 8001 | `--embeddings --pooling last` |
| rerank | `mradermacher/Qwen3-Reranker-0.6B-GGUF:Q4_K_M` | 8002 | `--reranking` |

Override any of them with `SPARKY_CHAT_GGUF`, `SPARKY_EMBED_GGUF`, `SPARKY_RERANK_GGUF`.

## Local

```bash
just model     # starts all three; GGUFs download on first run into the modelcache volume
```

Endpoints: `http://localhost:8000/v1` (chat), `:8001/v1` (embeddings), `:8002/v1/rerank`.
From inside compose the hosts are `chat`, `embed`, and `rerank`.

On a 6 GB card all three fit with about 400 MB to spare: embed and rerank together take
~2.2 GB, chat takes ~3.1 GB (2.5 GB of weights plus a 576 MB KV cache at 4096 context).
The KV cache is what runs you out of VRAM first, so raise `SPARKY_CHAT_CTX` only if there is
headroom; lower `--n-gpu-layers` to spill layers to CPU on a smaller GPU.

Rerank scores are uncalibrated — Qwen3-Reranker emits yes/no logits, so values are tiny
(1e-08 range) and only the ordering is meaningful. Rank by them; do not threshold on them.

## Deployed

Same three services on a GPU host with a larger chat GGUF. `--ctx-size` and the number of
parallel slots (`-np`) set the KV cache, which is what actually consumes VRAM under load;
size the GPU for `ctx-size x slots`, not for the weights alone.
