# deploy/inference

Model serving: `llama-server` from llama.cpp, configured here. Every model speaks the
OpenAI-compatible API, the only surface `apps/engine` and `apps/scraper` use.

`llama-server` serves one model per process, so each role is its own container.

| Role | Default GGUF | Port | Flags |
|---|---|---|---|
| chat | `Qwen/Qwen3-4B-GGUF:Q4_K_M` | 8000 | `--jinja` for tool calls, `--ctx-size 4096` |
| embeddings | `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` | 8001 | `--embeddings --pooling last`, batch 1024 |

Override either with `SPARKY_CHAT_GGUF` or `SPARKY_EMBED_GGUF`.

## Local

```bash
just model     # starts both; GGUFs download on first run into the modelcache volume
```

Endpoints: `http://localhost:8000/v1` (chat) and `:8001/v1` (embeddings). From inside compose
the hosts are `chat` and `embed`.

On a 6 GB card both fit with room to spare: embed takes ~0.9 GB, chat ~3.1 GB (2.5 GB of
weights plus a 576 MB KV cache at 4096 context). The KV cache grows with context; raise
`SPARKY_CHAT_CTX` only with headroom, and lower `--n-gpu-layers` to spill to CPU.

## Deployed

Same two services on a GPU host with a larger chat GGUF. `--ctx-size` and the number of
parallel slots (`-np`) set the KV cache size. Size the GPU for `ctx-size x slots` plus the
weights.
