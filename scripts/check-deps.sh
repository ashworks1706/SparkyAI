#!/usr/bin/env bash
# Rust apps don't depend on each other; discord and cli talk to engine over HTTP.
set -euo pipefail
meta=$(cargo metadata --format-version 1 --no-deps)
bad=$(jq -r '.packages[] | .name as $n | .dependencies[] | select(.path != null) | "\($n) -> \(.name)"' <<<"$meta")
if [ -n "$bad" ]; then echo "FAIL in-repo dependency: $bad"; exit 1; fi
echo "dependency direction ok"
