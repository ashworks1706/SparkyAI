#!/usr/bin/env bash
# Fetches the upstream PostHog files the posthog compose profile mounts.
# Sparse checkout of the pinned commit into $SPARKY_POSTHOG_DIR/src. Idempotent: a second run
# with the same pin does nothing.
# Usage: scripts/posthog.sh
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
