use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use proc_macro2::TokenStream;

mod cases;
mod ir_behavior_cases;
mod ir_opt_cases;
mod ir_postreg_cases;

struct Case {
    name: &'static str,
    ty: TokenStream,
    values: Vec<TokenStream>,
    inputs: Vec<DecodeInput>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WireFormat {
    Json,
    Postcard,
}

struct DecodeInput {
    name: &'static str,
    format: WireFormat,
    input: &'static [u8],
    expect: DecodeExpectation,
}

enum DecodeExpectation {
    Ok(TokenStream),
    Err(TokenStream),
    AnyErr,
}

struct CaseBuilder {
    case: Case,
}

impl CaseBuilder {
    fn new(name: &'static str, ty: TokenStream) -> Self {
        Self {
            case: Case {
                name,
                ty,
                values: Vec::new(),
                inputs: Vec::new(),
            },
        }
    }

    fn input_ok(
        mut self,
        format: WireFormat,
        name: &'static str,
        input: &'static [u8],
        expected: TokenStream,
    ) -> Self {
        self.case.inputs.push(DecodeInput {
            name,
            format,
            input,
            expect: DecodeExpectation::Ok(expected),
        });
        self
    }

    fn input_err(
        mut self,
        format: WireFormat,
        name: &'static str,
        input: &'static [u8],
        err_code: TokenStream,
    ) -> Self {
        self.case.inputs.push(DecodeInput {
            name,
            format,
            input,
            expect: DecodeExpectation::Err(err_code),
        });
        self
    }

    fn json_ok(self, name: &'static str, input: &'static str, expected: TokenStream) -> Self {
        self.input_ok(WireFormat::Json, name, input.as_bytes(), expected)
    }

    fn json_err(self, name: &'static str, input: &'static str, err_code: TokenStream) -> Self {
        self.input_err(WireFormat::Json, name, input.as_bytes(), err_code)
    }

    fn json_err_any(mut self, name: &'static str, input: &'static str) -> Self {
        self.case.inputs.push(DecodeInput {
            name,
            format: WireFormat::Json,
            input: input.as_bytes(),
            expect: DecodeExpectation::AnyErr,
        });
        self
    }

    fn build(self) -> Case {
        self.case
    }
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
    let command = args.next();
    let command_args: Vec<String> = args.collect();

    match command.as_deref() {
        Some("install") => install(),
        Some("gen") => {
            generate_synthetic();
            generate_ir_behavior_corpus();
            generate_ir_opt_corpus();
            generate_ir_postreg_corpus();
        }
        Some("test-x86_64") => test_x86_64(&command_args),
        _ => {
            eprintln!(
                "usage: cargo run --manifest-path xtask/Cargo.toml -- <install|gen|test-x86_64 [--full] [-- <extra nextest args...>]>"
            );
            std::process::exit(2);
        }
    }
}

fn install() {
    let root = workspace_root();
    let package = "kajit-mir-mcp";
    let binary = platform_binary_name(package);

    println!("building {package} in release mode...");
    let status = Command::new("cargo")
        .args(["build", "--release", "-p", package])
        .current_dir(&root)
        .status()
        .expect("failed to run cargo build");
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }

    let src = root.join("target").join("release").join(&binary);
    if !src.exists() {
        eprintln!("build finished but binary not found at {}", src.display());
        std::process::exit(1);
    }

    let dst_dir = cargo_bin_dir();
    if let Err(err) = fs::create_dir_all(&dst_dir) {
        eprintln!(
            "failed to create cargo bin directory {}: {err}",
            dst_dir.display()
        );
        std::process::exit(1);
    }

    let dst = dst_dir.join(&binary);
    if let Err(err) = fs::copy(&src, &dst) {
        eprintln!(
            "failed to copy {} to {}: {err}",
            src.display(),
            dst.display()
        );
        std::process::exit(1);
    }
    println!("copied {package} to {}", dst.display());

    #[cfg(target_os = "macos")]
    {
        println!("codesigning installed binary...");
        let status = Command::new("codesign")
            .args(["--sign", "-", "--force"])
            .arg(&dst)
            .status()
            .expect("failed to run codesign");
        if !status.success() {
            eprintln!("warning: codesign failed, continuing anyway");
        } else {
            let verify = Command::new("codesign")
                .args(["--verify", "--verbose=2"])
                .arg(&dst)
                .status()
                .expect("failed to run codesign --verify");
            if !verify.success() {
                eprintln!("warning: codesign verification failed, continuing anyway");
            }
        }
    }

    println!("validating installed binary...");
    let output = Command::new(&dst)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|err| {
            eprintln!("failed to execute {}: {err}", dst.display());
            std::process::exit(1);
        });
    if !output.status.success() {
        eprintln!(
            "installed binary exited with {} while validating",
            output.status
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            eprintln!("stderr:\n{stderr}");
        }
        std::process::exit(output.status.code().unwrap_or(1));
    }
    println!("installed and validated: {}", dst.display());
}

fn benches_prefix() -> PathBuf {
    workspace_root().join("kajit").join("benches")
}

fn tests_prefix() -> PathBuf {
    workspace_root().join("kajit").join("tests")
}

fn generate_synthetic() {
    let bench_path = benches_prefix().join("synthetic.rs");
    let test_path = tests_prefix().join("corpus.rs");
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

fn platform_binary_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    }
}

fn cargo_bin_dir() -> PathBuf {
    if let Some(cargo_home) = std::env::var_os("CARGO_HOME") {
        let cargo_home = PathBuf::from(cargo_home);
        if cargo_home.is_absolute() {
            return cargo_home.join("bin");
        }
        return home_dir().join(cargo_home).join("bin");
    }
    home_dir().join(".cargo").join("bin")
}

fn home_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home);
    }
    if let Some(home) = std::env::var_os("USERPROFILE") {
        return PathBuf::from(home);
    }
    panic!("unable to determine home directory (HOME/USERPROFILE are unset)");
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent directory");
    }
    fs::write(path, content).expect("write generated file");
}

const X86_64_REGRESSION_TESTS: &[&str] = &[
    "prop::deny_unknown_fields",
    "prop::flat_struct",
    "prop::scalar_i64",
    "prop::nested_struct",
    "prop::transparent_composite",
    "prop::shared_inner_type",
];

fn test_x86_64(args: &[String]) {
    if !cfg!(target_os = "macos") {
        eprintln!("`cargo xtask test-x86_64` is only supported on macOS hosts.");
        std::process::exit(2);
    }

    ensure_rosetta_available();
    ensure_x86_64_target_installed();

    let mut nextest_args = vec![
        "nextest".to_string(),
        "run".to_string(),
        "-p".to_string(),
        "kajit".to_string(),
        "--target".to_string(),
        "x86_64-apple-darwin".to_string(),
    ];

    let mut run_full = false;
    let mut passthrough_start = None;
    for (idx, arg) in args.iter().enumerate() {
        if arg == "--full" {
            run_full = true;
        } else if arg == "--" {
            passthrough_start = Some(idx + 1);
            break;
        } else {
            eprintln!("unknown argument `{arg}`");
            eprintln!("usage: cargo xtask test-x86_64 [--full] [-- <extra nextest args...>]");
            std::process::exit(2);
        }
    }

    if !run_full {
        nextest_args.extend(X86_64_REGRESSION_TESTS.iter().map(|t| (*t).to_string()));
    }

    if let Some(start) = passthrough_start {
        nextest_args.extend(args[start..].iter().cloned());
    }

    println!("running: cargo {}", nextest_args.join(" "));
    let status = Command::new("cargo")
        .args(nextest_args)
        .status()
        .expect("failed to run cargo nextest");

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn ensure_rosetta_available() {
    let status = Command::new("arch")
        .args(["-x86_64", "/usr/bin/true"])
        .status()
        .expect("failed to check Rosetta availability");

    if !status.success() {
        eprintln!("Rosetta is required for x86_64 test execution on Apple Silicon.");
        eprintln!("Install it with: softwareupdate --install-rosetta --agree-to-license");
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn ensure_x86_64_target_installed() {
    let output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .expect("failed to query installed Rust targets");

    if !output.status.success() {
        eprintln!("failed to read installed Rust targets via rustup");
        std::process::exit(output.status.code().unwrap_or(1));
    }

    let installed = String::from_utf8(output.stdout).expect("rustup output should be utf-8");
    if !installed
        .lines()
        .any(|line| line.trim() == "x86_64-apple-darwin")
    {
        eprintln!("missing Rust target `x86_64-apple-darwin`.");
        eprintln!("Install it with: rustup target add x86_64-apple-darwin");
        std::process::exit(2);
    }
}
