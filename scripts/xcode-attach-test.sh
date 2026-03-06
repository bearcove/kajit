#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <test_name>"
  echo "example: $0 json::bool_true_false"
  exit 1
fi

filter="$1"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"

mktemp_portable() {
  local prefix="$1"
  mktemp -t "${prefix}.XXXXXX"
}

json_output="$(cargo nextest list -p kajit -E "test(=$filter)" --message-format json)"

match_json=""
if match_json="$(python3 -c '
import json
import sys

matches = []
for raw in sys.stdin:
    raw = raw.strip()
    if not raw.startswith("{"):
        continue
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        continue
    suites = msg.get("rust-suites")
    if not isinstance(suites, dict):
        continue
    for suite in suites.values():
        bin_path = suite.get("binary-path")
        testcases = suite.get("testcases", {})
        if not isinstance(testcases, dict):
            continue
        for test_name, testcase in testcases.items():
            fm = testcase.get("filter-match", {})
            if isinstance(fm, dict) and fm.get("status") == "matches":
                matches.append((bin_path, test_name))

if not matches:
    sys.exit(2)

uniq = []
seen = set()
for pair in matches:
    if pair in seen:
        continue
    seen.add(pair)
    uniq.append(pair)

if len(uniq) != 1:
    print(json.dumps({"error": "ambiguous", "matches": uniq}))
    sys.exit(3)

bin_path, test_name = uniq[0]
print(json.dumps({"binary_path": bin_path, "test_name": test_name}))
')"; then
  :
else
  status=$?
  if [[ $status -eq 2 ]]; then
    echo "no matching test found for filter: $filter"
  elif [[ $status -eq 3 ]]; then
    echo "filter matched more than one test; use a more specific name: $filter"
  else
    echo "failed to resolve a unique test for filter: $filter"
  fi
  exit 1
fi <<<"$json_output"

binary_path="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["binary_path"])' <<<"$match_json")"
test_name="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["test_name"])' <<<"$match_json")"

cfg_mir_tmp="$(mktemp_portable kajit-xcode-cfg)"
ref_tmp="$(mktemp_portable kajit-xcode-ref)"
lldb_tmp="$(mktemp_portable kajit-xcode-lldb)"
cleanup() {
  rm -f "$cfg_mir_tmp" "$ref_tmp" "$lldb_tmp"
}
trap cleanup EXIT

have_ref=0
if cargo run -q --manifest-path "$repo_root/xtask/Cargo.toml" -- corpus-cfg-mir "$filter" >"$cfg_mir_tmp" 2>/dev/null; then
  if cargo run -q --manifest-path "$repo_root/xtask/Cargo.toml" -- \
    debug-cfg-mir lldb-ref "$cfg_mir_tmp" --corpus-test "$filter" >"$ref_tmp" 2>/dev/null; then
    have_ref=1
  fi
fi

{
  echo "settings set plugin.jit-loader.gdb.enable on"
  if [[ $have_ref -eq 1 ]]; then
    printf 'script import os; os.environ["KAJIT_LLDB_REF_FILE"] = r"%s"\n' "$ref_tmp"
  fi
  printf 'command script import %s/scripts/kajit_lldb_side_by_side.py\n' "$repo_root"
  echo "breakpoint set -n __jit_debug_register_code"
} >"$lldb_tmp"

echo "KAJIT_DEBUG=1"
echo "KAJIT_WAIT_FOR_DEBUGGER=1"
echo "binary: $binary_path"
echo "test:   $test_name"
echo "lldb setup file: $lldb_tmp"
if [[ $have_ref -eq 1 ]]; then
  echo "side-by-side reference: $ref_tmp"
fi

KAJIT_DEBUG=1 KAJIT_WAIT_FOR_DEBUGGER=1 "$binary_path" --exact "$test_name" --nocapture &
child_pid=$!

echo
echo "pid: $child_pid"
echo "1. In Xcode: Debug > Attach to Process by PID or Name... -> $child_pid"
echo "2. In the Xcode debug console: command source $lldb_tmp"
echo "3. Press Continue. The process will resume, hit __jit_debug_register_code, and then you can run kajit-break."
echo

wait "$child_pid"
