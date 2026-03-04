#!/usr/bin/env bash
# nextest wrapper script: run test binary under LLDB with JIT DWARF support.
#
# Usage: cargo nextest run --profile lldb -E 'test(=json::bool_true_false)'

export KAJIT_DEBUG=1

exec lldb \
  -o 'settings set plugin.jit-loader.gdb.enable on' \
  -o 'breakpoint set -n __jit_debug_register_code' \
  -o 'run' \
  -- "$@"
