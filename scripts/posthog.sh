#!/usr/bin/env bash
# Fetches the upstream PostHog files the posthog compose profile mounts.
# Sparse checkout of the pinned commit into $SPARKY_POSTHOG_DIR/src and the GeoIP database
# into $SPARKY_POSTHOG_DIR/share. Idempotent: a second run with the same pin does nothing.
# Usage: scripts/posthog.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
sha="$(tr -d '[:space:]' < "$root/deploy/posthog/VERSION")"
dir="${SPARKY_POSTHOG_DIR:-$root/.sparky/posthog}"
src="$dir/src"
share="$dir/share"
repo="https://github.com/PostHog/posthog.git"

paths=(
  posthog/idl
  posthog/user_scripts
  docker/clickhouse
  docker/temporal/dynamicconfig
  docker/kafka/topics.txt
  docker/livestream/configs-hobby.yml
)

mkdir -p "$dir" "$share"

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

if [ ! -s "$share/GeoLite2-City.mmdb" ]; then
  command -v brotli >/dev/null || { echo "posthog: brotli is required to unpack GeoIP" >&2; exit 1; }
  tmp="$(mktemp)"
  curl -fsSL --http1.1 https://mmdbcdn.posthog.net/ | brotli --decompress > "$tmp"
  mv "$tmp" "$share/GeoLite2-City.mmdb"
  chmod 644 "$share/GeoLite2-City.mmdb"
  printf '{"date": "%s"}\n' "$(date +%Y-%m-%d)" > "$share/GeoLite2-City.json"
  echo "posthog: downloaded GeoIP into $share"
else
  echo "posthog: GeoIP present in $share"
fi
