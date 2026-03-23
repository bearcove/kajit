image_name := "kajit-test-x86_64"
docker_platform := "linux/amd64"

list:
    just --list

# Build the x86_64 Docker test image
docker-build:
    docker build --platform {{ docker_platform }} -t {{ image_name }} .

# Run tests in x86_64 Docker container
docker-test: docker-build
    docker run --platform {{ docker_platform }} --rm -v {{ justfile_directory() }}:/workspace -w /workspace {{ image_name }} cargo nextest run

# Run a shell in the x86_64 Docker container
docker-shell: docker-build
    docker run --platform {{ docker_platform }} --rm -it -v {{ justfile_directory() }}:/workspace -w /workspace {{ image_name }} bash

# Run benchmarks in x86_64 Docker container
docker-bench: docker-build
    docker run --platform {{ docker_platform }} --rm -v {{ justfile_directory() }}:/workspace -w /workspace {{ image_name }} cargo bench

# Run tests natively
test:
    cargo nextest run

# Refresh disassembly snapshots for host architecture
snapshots-native:
    cargo insta test --accept --test-runner nextest -- disasm_

# Refresh disassembly snapshots for x86_64 via Docker
snapshots-x86_64: docker-build
    docker run --platform {{ docker_platform }} --rm -v {{ justfile_directory() }}:/workspace -w /workspace {{ image_name }} cargo insta test --accept --test-runner nextest -- disasm_

# Refresh disassembly snapshots for both supported architectures
snapshots-all: snapshots-native snapshots-x86_64

# Run benchmarks natively
bench *args:
    cargo bench {{ args }}

# Run all benchmarks and generate bench_report/index.html
bench-report *args:
    cargo bench {{ args }} | cargo run --example bench_report

# Check + clippy
check:
    cargo check
    cargo clippy -- -D warnings
