#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace_root"

usage() {
  cat <<'EOF'
Run a pass-combination matrix against a focused nextest expression.

Usage:
  scripts/opts-matrix.sh [options] [-- <extra cargo-nextest args...>]

Options:
  --expr <expr>            nextest expression (default: test(=json::all_scalars))
  --package <name>         cargo package (default: kajit)
  --test <name>            integration test binary (default: corpus)
  --pass <name>            pass to include in matrix (repeatable).
                           If omitted, uses the default 4 passes.
                           Non-selected default passes are forced off.
  --regalloc <on|off>      set +regalloc or -regalloc base toggle (default: on)
  --log-dir <path>         parent directory for run logs
                           (default: target/opts-matrix-logs)
  --keep-passing-logs      keep logs for passing combinations (default: false)
  -h, --help               show this help text

Examples:
  scripts/opts-matrix.sh
  scripts/opts-matrix.sh --pass theta_loop_invariant_hoist --pass inline_apply
  scripts/opts-matrix.sh --expr 'test(=json::all_scalars)' -- --no-fail-fast
EOF
}

expr="test(=json::all_scalars)"
package="kajit"
test_bin="corpus"
regalloc="on"
keep_passing_logs=false
pass_list=()
custom_passes=false
extra_nextest_args=()
log_root="$workspace_root/target/opts-matrix-logs"

while (($# > 0)); do
  case "$1" in
    --expr)
      expr="$2"
      shift 2
      ;;
    --package)
      package="$2"
      shift 2
      ;;
    --test)
      test_bin="$2"
      shift 2
      ;;
    --pass)
      if ! $custom_passes; then
        pass_list=()
        custom_passes=true
      fi
      pass_list+=("$2")
      shift 2
      ;;
    --regalloc)
      regalloc="$2"
      if [[ "$regalloc" != "on" && "$regalloc" != "off" ]]; then
        echo "invalid --regalloc value: $regalloc (expected on|off)" >&2
        exit 2
      fi
      shift 2
      ;;
    --log-dir)
      log_root="$2"
      shift 2
      ;;
    --keep-passing-logs)
      keep_passing_logs=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      extra_nextest_args=("$@")
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

default_passes=(
  "bounds_check_coalescing"
  "theta_loop_invariant_hoist"
  "inline_apply"
  "dead_code_elimination"
)

if ((${#pass_list[@]} == 0)); then
  pass_list=("${default_passes[@]}")
fi

all_known_passes=("${default_passes[@]}")
for pass in "${pass_list[@]}"; do
  seen=false
  for known in "${all_known_passes[@]}"; do
    if [[ "$known" == "$pass" ]]; then
      seen=true
      break
    fi
  done
  if ! $seen; then
    all_known_passes+=("$pass")
  fi
done

pass_count=${#pass_list[@]}
if ((pass_count == 0)); then
  echo "no passes configured" >&2
  exit 2
fi
if ((pass_count > 20)); then
  echo "refusing to run 2^${pass_count} combinations; pass count too large" >&2
  exit 2
fi

total=$((1 << pass_count))
run_id="$(date +%Y%m%d-%H%M%S)"
run_dir="$log_root/$run_id"
mkdir -p "$run_dir"

regalloc_token="+regalloc"
if [[ "$regalloc" == "off" ]]; then
  regalloc_token="-regalloc"
fi

echo "opts-matrix run: $run_id"
echo "  package/test: $package/$test_bin"
echo "  expr: $expr"
echo "  passes ($pass_count): ${pass_list[*]}"
echo "  base opts: +all_opts,$regalloc_token (explicit +/-pass toggles)"
echo "  logs: $run_dir"
echo
printf "%-8s %-6s %-40s %s\n" "mask" "result" "enabled_passes" "KAJIT_OPTS"

pass_count_u32=$pass_count
failed=0

for ((mask = 0; mask < total; mask++)); do
  bits=""
  enabled_passes=()
  opts_tokens=("+all_opts" "$regalloc_token")

  # Start from a known baseline where every tracked pass is disabled,
  # then enable only the passes selected by the bitmask.
  for pass in "${all_known_passes[@]}"; do
    opts_tokens+=("-pass.$pass")
  done

  for ((i = 0; i < pass_count_u32; i++)); do
    bit=$(((mask >> i) & 1))
    if ((bit == 1)); then
      pass="${pass_list[$i]}"
      enabled_passes+=("$pass")
      opts_tokens+=("+pass.$pass")
      bits="1$bits"
    else
      bits="0$bits"
    fi
  done

  if ((${#enabled_passes[@]} == 0)); then
    enabled_label="none"
  else
    enabled_label="$(IFS=+; echo "${enabled_passes[*]}")"
  fi

  opts_value="$(IFS=,; echo "${opts_tokens[*]}")"
  safe_label="${enabled_label//+/_}"
  safe_label="${safe_label//\//_}"
  log_file="$run_dir/${bits}_${safe_label}.log"

  cmd=(cargo nextest run -p "$package" --test "$test_bin" -E "$expr")
  if ((${#extra_nextest_args[@]} > 0)); then
    cmd+=("${extra_nextest_args[@]}")
  fi

  if KAJIT_OPTS="$opts_value" "${cmd[@]}" >"$log_file" 2>&1; then
    result="PASS"
    if ! $keep_passing_logs; then
      rm -f "$log_file"
    fi
  else
    result="FAIL"
    failed=$((failed + 1))
  fi

  printf "%-8s %-6s %-40s %s\n" "$bits" "$result" "$enabled_label" "$opts_value"
done

echo
if ((failed == 0)); then
  echo "all $total combinations passed"
else
  echo "$failed / $total combinations failed"
  echo "failure logs kept in: $run_dir"
fi
