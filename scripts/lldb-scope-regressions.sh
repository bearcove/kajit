#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

echo "running LLDB scope regressions"

bash scripts/lldb-check-vars.sh json::bool_true_false \
  4:v46:unavailable \
  5:v46:available \
  5:v46:listed \
  5:v10:unlisted \
  24:v10:listed \
  24:v20:listed \
  24:v21:listed \
  24:v23:listed \
  24:v33:listed

echo "LLDB scope regressions passed"
