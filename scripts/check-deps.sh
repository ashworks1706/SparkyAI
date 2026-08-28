#!/usr/bin/env bash
# Enforces crate dependency direction from docs/ARCHITECTURE.md:
#   harness  -> no in-repo deps
#   adapters -> harness only
#   app      -> anything
set -euo pipefail
meta=$(cargo metadata --format-version 1 --no-deps)
fail=0
for pkg in $(jq -r '.packages[].name' <<<"$meta"); do
  deps=$(jq -r --arg p "$pkg" '.packages[] | select(.name==$p) | .dependencies[].name | select(startswith("sparky-"))' <<<"$meta" | sort | tr '\n' ' ')
  case "$pkg" in
    sparky-harness) [ -z "$deps" ] || { echo "FAIL $pkg depends on: $deps (must be none)"; fail=1; } ;;
    sparky-app) ;;
    *) for d in $deps; do [ "$d" = "sparky-harness" ] || { echo "FAIL $pkg depends on $d (adapters may only depend on sparky-harness)"; fail=1; }; done ;;
  esac
done
[ $fail -eq 0 ] && echo "dependency direction ok"
exit $fail
