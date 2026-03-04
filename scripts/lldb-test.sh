#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <test_name>"
  echo "example: $0 json::bool_true_false"
  exit 1
fi

filter="$1"
export KAJIT_DEBUG=1

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

echo "KAJIT_DEBUG=1"
echo "binary: $binary_path"
echo "test:   $test_name"
echo "LLDB will start without auto-run. Type 'run' when ready."

exec lldb \
  -o 'settings set plugin.jit-loader.gdb.enable on' \
  -o 'breakpoint set -n __jit_debug_register_code' \
  -- "$binary_path" --exact "$test_name" --nocapture
