use std::fs;
use std::path::{Path, PathBuf};

use proc_macro2::TokenStream;

mod cases;
mod ir_behavior_cases;
mod ir_opt_cases;
mod ir_postreg_cases;

struct Case {
    name: &'static str,
    values: Vec<TokenStream>,
}

struct IrOptCase {
    name: &'static str,
    ir: &'static str,
    must_not_contain_after: &'static [&'static str],
    must_contain_after: &'static [&'static str],
    input: &'static [u8],
}

struct IrPostRegCase {
    name: &'static str,
    ir: &'static str,
    must_contain_linear: &'static [&'static str],
    max_total_edits: usize,
    input: &'static [u8],
    expected: u8,
}

struct BehaviorVector {
    input: &'static [u8],
    expected: u8,
}

struct IrBehaviorCase {
    name: &'static str,
    ir: &'static str,
    vectors: &'static [BehaviorVector],
}

fn main() {
    let mut args = std::env::args();
    let _bin = args.next();
    match args.next().as_deref() {
        Some("gen") => {
            generate_synthetic();
            generate_ir_behavior_corpus();
            generate_ir_opt_corpus();
            generate_ir_postreg_corpus();
        }
        _ => {
            eprintln!(
                "usage: cargo run --manifest-path xtask/Cargo.toml -- <generate-synthetic|generate-ir-opt-corpus|generate-ir-postreg-corpus|generate-ir-behavior-corpus>"
            );
            std::process::exit(2);
        }
    }
}

fn benches_prefix() -> PathBuf {
    workspace_root().join("kajit").join("benches")
}

fn tests_prefix() -> PathBuf {
    workspace_root().join("kajit").join("tests")
}

fn generate_synthetic() {
    let bench_path = benches_prefix().join("synthetic.rs");
    let test_path = tests_prefix().join("generated_synthetic.rs");
    write_file(&bench_path, &cases::render_bench_file());
    write_file(&test_path, &cases::render_test_file());
    println!(
        "generated:\n- {}\n- {}",
        bench_path.display(),
        test_path.display()
    );
}

fn generate_ir_opt_corpus() {
    let test_path = tests_prefix().join("generated_ir_opt_corpus.rs");
    write_file(&test_path, &ir_opt_cases::render_ir_opt_test_file());
    println!("generated:\n- {}", test_path.display());
}

fn generate_ir_postreg_corpus() {
    let test_path = tests_prefix().join("generated_ir_postreg_corpus.rs");
    write_file(&test_path, &ir_postreg_cases::render_ir_postreg_test_file());
    println!("generated:\n- {}", test_path.display());
}

fn generate_ir_behavior_corpus() {
    let test_path = tests_prefix().join("generated_ir_behavior_corpus.rs");
    write_file(
        &test_path,
        &ir_behavior_cases::render_ir_behavior_test_file(),
    );
    println!("generated:\n- {}", test_path.display());
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask should live in workspace/xtask")
        .to_path_buf()
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent directory");
    }
    fs::write(path, content).expect("write generated file");
}
