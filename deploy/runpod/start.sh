#!/usr/bin/env bash
# Entrypoint for a RunPod pod running the vllm/vllm-openai image.
# Reads MODEL / VLLM_ARGS / VLLM_API_KEY from the pod's environment.
set -euo pipefail
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --api-key "${VLLM_API_KEY}" \
  ${VLLM_ARGS:-}
