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
  24:v33:listed \
  30:key_ptr:listed \
  30:key_len:listed \
  30:is_field_a:listed \
  30:v11:unlisted \
  31:is_field_b:listed \
  31:v22:unlisted \
  33:handled_field:listed \
  38:a:listed \
  38:a:unavailable \
  39:a:available

echo "LLDB scope regressions passed"
