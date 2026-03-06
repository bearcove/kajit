#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <test_name> <line:var:available|unavailable|listed|unlisted>..."
  echo "example: $0 json::bool_true_false 4:v46:unavailable 5:v46:available 5:v47:unlisted"
  exit 1
fi

filter="$1"
shift
expectations=("$@")
export KAJIT_DEBUG=1
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
break_regex="${KAJIT_LLDB_BREAK_REGEX:-kajit::decode::}"

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

cfg_mir_tmp="$(mktemp_portable kajit-lldb-cfg)"
ref_tmp="$(mktemp_portable kajit-lldb-ref)"
output_tmp="$(mktemp_portable kajit-lldb-out)"
cleanup() {
  rm -f "$cfg_mir_tmp" "$ref_tmp" "$output_tmp"
}
trap cleanup EXIT

if cargo run -q --manifest-path "$repo_root/xtask/Cargo.toml" -- corpus-cfg-mir "$filter" >"$cfg_mir_tmp" 2>/dev/null; then
  if cargo run -q --manifest-path "$repo_root/xtask/Cargo.toml" -- \
    debug-cfg-mir lldb-ref "$cfg_mir_tmp" --corpus-test "$filter" >"$ref_tmp" 2>/dev/null; then
    export KAJIT_LLDB_REF_FILE="$ref_tmp"
  fi
fi

lldb_args=(
  --batch
  -o "settings set plugin.jit-loader.gdb.enable on"
  -o "command script import $repo_root/scripts/kajit_lldb_side_by_side.py"
  -o "breakpoint set -n __jit_debug_register_code"
  -o "run"
  -o "kajit-break $break_regex"
  -o "continue"
)

for spec in "${expectations[@]}"; do
  IFS=: read -r line var state extra <<<"$spec"
  if [[ -n "${extra:-}" || -z "${line:-}" || -z "${var:-}" || -z "${state:-}" ]]; then
    echo "invalid expectation '$spec' (expected line:var:available|unavailable|listed|unlisted)"
    exit 1
  fi
  if [[ "$state" != "available" && "$state" != "unavailable" && "$state" != "listed" && "$state" != "unlisted" ]]; then
    echo "invalid state '$state' in expectation '$spec'"
    exit 1
  fi
  lldb_args+=(-o "kajit-expect $line $var $state")
done

echo "KAJIT_DEBUG=1"
echo "binary: $binary_path"
echo "test:   $test_name"
echo "regex:  $break_regex"
printf 'check:  %s\n' "${expectations[@]}"

set +e
lldb "${lldb_args[@]}" -- "$binary_path" --exact "$test_name" --nocapture >"$output_tmp" 2>&1
lldb_status=$?
set -e

cat "$output_tmp"

if [[ $lldb_status -ne 0 ]]; then
  echo "lldb exited with status $lldb_status"
  exit "$lldb_status"
fi

fail_count="$(grep -c '^FAIL ' "$output_tmp" || true)"
ok_count="$(grep -c '^OK line=' "$output_tmp" || true)"
expected_count="${#expectations[@]}"

if [[ "$fail_count" -ne 0 ]]; then
  echo "scripted LLDB checks failed"
  exit 1
fi

if [[ "$ok_count" -ne "$expected_count" ]]; then
  echo "scripted LLDB checks were incomplete: expected $expected_count OK lines, saw $ok_count"
  exit 1
fi

echo "scripted LLDB checks passed"
