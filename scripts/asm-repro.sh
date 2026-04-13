#!/bin/bash
set -euxo pipefail

export CARGO_TARGET_DIR=/tmp/kajit-asan
export RUSTFLAGS="-Z sanitizer=address"
export RUST_LOG=trace

cargo +nightly nextest run \
    --target aarch64-apple-darwin \
    --features facet-styx/tracing \
    --no-capture "${ASM_REPRO_TEST:-asm_repro_schema_loads}"
