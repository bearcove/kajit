#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <test_name> [--no-open]"
  echo "example: $0 json::bool_true_false"
  exit 1
fi

filter="$1"
open_project=1
if [[ $# -eq 2 ]]; then
  if [[ "$2" != "--no-open" ]]; then
    echo "unknown option: $2"
    exit 1
  fi
  open_project=0
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

mktemp_portable() {
  local prefix="$1"
  mktemp -d -t "${prefix}.XXXXXX"
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

cfg_mir_tmp="$(mktemp -t kajit-xcode-cfg.XXXXXX)"
ref_tmp="$(mktemp -t kajit-xcode-ref.XXXXXX)"
have_ref=0
if cargo run -q --manifest-path "$repo_root/xtask/Cargo.toml" -- corpus-cfg-mir "$filter" >"$cfg_mir_tmp" 2>/dev/null; then
  if cargo run -q --manifest-path "$repo_root/xtask/Cargo.toml" -- \
    debug-cfg-mir lldb-ref "$cfg_mir_tmp" --corpus-test "$filter" >"$ref_tmp" 2>/dev/null; then
    have_ref=1
  fi
fi
rm -f "$cfg_mir_tmp"

project_root="$(mktemp_portable kajit-xcode-project)"
project_name="KajitDebug_${test_name//[:]/_}"
project_path="$project_root/$project_name.xcodeproj"
sources_dir="$project_root/Sources"
mkdir -p "$sources_dir"

python3 - <<'PY' "$binary_path" "$test_name" "$sources_dir/main.swift"
import json
import pathlib
import sys

binary_path = sys.argv[1]
test_name = sys.argv[2]
out_path = pathlib.Path(sys.argv[3])

swift = f"""import Foundation

let binaryPath = {json.dumps(binary_path)}
let args = ["--exact", {json.dumps(test_name)}, "--nocapture"]
var env = ProcessInfo.processInfo.environment
env["KAJIT_DEBUG"] = "1"

var argv = [strdup(binaryPath)]
argv.append(contentsOf: args.map {{ strdup($0) }})
argv.append(nil)

var envPairs = env.map {{ strdup("\\($0.key)=\\($0.value)") }}
envPairs.append(nil)

execve(binaryPath, argv, envPairs)

perror("execve")
exit(1)
"""

out_path.write_text(swift, encoding="utf-8")
PY

cat >"$project_root/project.yml" <<YAML
name: $project_name
options:
  xcodeVersion: "16.0"
targets:
  $project_name:
    type: tool
    platform: macOS
    deploymentTarget: "14.0"
    sources:
      - path: Sources
    scheme: {}
YAML

lldb_init="$project_root/kajit-xcode.lldbinit"
{
  echo "settings set plugin.jit-loader.gdb.enable on"
  if [[ $have_ref -eq 1 ]]; then
    printf 'script import os; os.environ["KAJIT_LLDB_REF_FILE"] = r"%s"\n' "$ref_tmp"
  fi
  printf 'command script import %s/scripts/kajit_lldb_side_by_side.py\n' "$repo_root"
  echo "breakpoint set -n __jit_debug_register_code"
} >"$lldb_init"

xcodegen generate --spec "$project_root/project.yml" --project "$project_root" >/dev/null

scheme_path="$project_path/xcshareddata/xcschemes/$project_name.xcscheme"
python3 - <<'PY' "$scheme_path" "$lldb_init"
import pathlib
import sys
import xml.etree.ElementTree as ET

scheme_path = pathlib.Path(sys.argv[1])
lldb_init = sys.argv[2]

tree = ET.parse(scheme_path)
root = tree.getroot()
for tag in ("TestAction", "LaunchAction"):
    node = root.find(tag)
    if node is not None:
        node.set("customLLDBInitFile", lldb_init)

tree.write(scheme_path, encoding="UTF-8", xml_declaration=True)
PY

echo "project: $project_path"
echo "binary:  $binary_path"
echo "test:    $test_name"
echo "lldb:    $lldb_init"
if [[ $have_ref -eq 1 ]]; then
  echo "ref:     $ref_tmp"
fi

if [[ $open_project -eq 1 ]]; then
  xed "$project_path"
fi
