# deploy/inference

Model serving. Not our code — vLLM on RunPod — but a deployable we own the configuration for.

| Pod | Env | GPU | Port |
|---|---|---|---|
| chat | `vllm.env` — Qwen3-14B, hermes tool parser | A100 80GB / H100 | 8000 |
| embed + rerank | `embedding.env` — Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B | L4 / A10 | 8001, 8002 |

`start.sh` is the pod start command. Point `SPARKY_MODEL__BASE_URL` etc. at `https://<pod-id>-<port>.proxy.runpod.net/v1`.

Later: a custom image serving Sparky LoRA adapters (`--enable-lora`).
