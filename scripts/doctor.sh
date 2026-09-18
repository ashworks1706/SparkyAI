#!/usr/bin/env bash
# Check required tools are installed and print versions.
set -uo pipefail
ok=0
need() { # Check tool presence and print version or hint
  if command -v "$2" >/dev/null 2>&1; then printf '  ok   %-8s %s\n' "$1" "$($2 --version 2>/dev/null | head -1)"; else printf '  MISSING %-8s install: %s\n' "$1" "$3"; ok=1; fi
}
echo "tools:"
need just    just    "https://just.systems"
need cargo   cargo   "https://rustup.rs"
need uv      uv      "https://docs.astral.sh/uv"
need node    node    "https://nodejs.org (22+)"
need npm     npm     "comes with node"
need docker  docker  "https://docs.docker.com/get-docker"
need jq      jq      "package manager"
echo "runtime:"
if docker version >/dev/null 2>&1; then
  printf '  ok   %-8s %s\n' "runtime" "run_sandbox has a container runtime"
else
  printf '  MISSING %-8s %s\n' "runtime" "run_sandbox needs one; set sandbox.required = false to run without it"
fi
echo "files:"
[ -f .env ] && echo "  ok   .env" || echo "  MISSING .env        run: just env"
[ "$(git config core.hooksPath)" = ".githooks" ] && echo "  ok   git hooks" || echo "  MISSING git hooks   run: just hooks"
exit $ok
