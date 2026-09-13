#!/usr/bin/env bash
# Sparse-checks out the pinned PostHog commit for the compose profile; safe to rerun.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
sha="$(tr -d '[:space:]' < "$root/deploy/posthog/VERSION")"
dir="${SPARKY_POSTHOG_DIR:-$root/.sparky/posthog}"
src="$dir/src"
repo="https://github.com/PostHog/posthog.git"

paths=(
  posthog/idl
  posthog/user_scripts
  docker/clickhouse
  docker/kafka/topics.txt
)

mkdir -p "$dir"

if [ "$(git -C "$src" rev-parse HEAD 2>/dev/null || true)" != "$sha" ]; then
  if [ ! -d "$src/.git" ]; then
    rm -rf "$src"
    git init -q "$src"
    git -C "$src" remote add origin "$repo"
  fi
  git -C "$src" sparse-checkout set --no-cone "${paths[@]/#//}"
  git -C "$src" fetch -q --depth 1 --filter=blob:none origin "$sha"
  git -C "$src" -c advice.detachedHead=false checkout -q --force FETCH_HEAD
  echo "posthog: checked out $sha into $src"
else
  echo "posthog: $src already at $sha"
fi
