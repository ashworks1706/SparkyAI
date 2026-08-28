#!/usr/bin/env bash
# Enforces dependency direction from docs/ARCHITECTURE.md:
#   crates/harness  -> no in-repo deps
#   crates/*        -> sparky-harness only   (runtime: none)
#   apps/api        -> anything in crates/
#   apps/discord    -> sparky-runtime only   (HTTP client of api; never links harness)
set -euo pipefail
meta=$(cargo metadata --format-version 1 --no-deps)
fail=0
for pkg in $(jq -r '.packages[].name' <<<"$meta"); do
  deps=$(jq -r --arg p "$pkg" '.packages[] | select(.name==$p) | .dependencies[].name | select(startswith("sparky-"))' <<<"$meta" | sort | tr '\n' ' ')
  case "$pkg" in
    sparky-harness|sparky-runtime) [ -z "$deps" ] || { echo "FAIL $pkg depends on: $deps (must be none)"; fail=1; } ;;
    sparky-api) ;;
    sparky-discord) [ "$deps" = "sparky-runtime " ] || { echo "FAIL $pkg depends on: $deps (only sparky-runtime allowed)"; fail=1; } ;;
    *) for d in $deps; do [ "$d" = "sparky-harness" ] || { echo "FAIL $pkg depends on $d (libraries may only depend on sparky-harness)"; fail=1; }; done ;;
  esac
done
[ $fail -eq 0 ] && echo "dependency direction ok"
exit $fail
