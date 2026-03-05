use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use proc_macro2::TokenStream;

mod cases;

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
        }
        Some("minimize-cfg-mir") => minimize_cfg_mir(&command_args),
        _ => {
            eprintln!(
                "usage: cargo run --manifest-path xtask/Cargo.toml -- <install|gen|minimize-cfg-mir>"
            );
            std::process::exit(2);
        }
    }
}

fn minimize_cfg_mir(args: &[String]) {
    let [path, input_hex] = args else {
        eprintln!(
            "usage: cargo run --manifest-path xtask/Cargo.toml -- minimize-cfg-mir <cfg-mir-path> <hex-input>"
        );
        std::process::exit(2);
    };

    let input = parse_hex_bytes(input_hex).unwrap_or_else(|err| {
        eprintln!("invalid hex input `{input_hex}`: {err}");
        std::process::exit(2);
    });
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        eprintln!("failed to read {}: {err}", path);
        std::process::exit(1);
    });
    let program = kajit_mir_text::parse_cfg_mir(&text).unwrap_or_else(|err| {
        eprintln!("failed to parse CFG-MIR from {}: {err}", path);
        std::process::exit(1);
    });
    let (reduced, stats, interestingness) =
        kajit_mir::minimize_cfg_program_for_differential(&program, &input).unwrap_or_else(|err| {
            match err {
                kajit_mir::MinimizeError::NotInteresting => {
                    eprintln!(
                        "seed program is not differentially interesting for input {input_hex}"
                    );
                }
                kajit_mir::MinimizeError::Predicate(message) => {
                    eprintln!("differential minimization failed: {message}");
                }
            }
            std::process::exit(1);
        });

    eprintln!(
        "reduced CFG-MIR for input {input_hex}: blocks {} -> {}, insts {} -> {}, edges {} -> {}, accepted {} / {}",
        stats.initial_size.blocks,
        stats.final_size.blocks,
        stats.initial_size.insts,
        stats.final_size.insts,
        stats.initial_size.edges,
        stats.final_size.edges,
        stats.accepted,
        stats.attempts
    );
    eprintln!(
        "preserved differential signature: field={}, ideal_trap={:?}, post_trap={:?}, ideal_returned={}, post_returned={}",
        interestingness.field,
        interestingness.ideal_trap,
        interestingness.post_trap,
        interestingness.ideal_returned,
        interestingness.post_returned
    );
    println!("{reduced}");
}

fn parse_hex_bytes(input: &str) -> Result<Vec<u8>, String> {
    let separators = ['[', ']', ',', ' ', '\t', '\n', '\r'];
    let has_separators = input.chars().any(|ch| separators.contains(&ch));

    if has_separators {
        let mut out = Vec::new();
        for raw in input.split(|ch| separators.contains(&ch)) {
            let token = raw.trim();
            if token.is_empty() {
                continue;
            }
            let token = token
                .strip_prefix("0x")
                .or_else(|| token.strip_prefix("0X"))
                .unwrap_or(token);
            if token.len() != 2 {
                return Err(format!(
                    "expected byte token with exactly 2 hex digits, got `{raw}`"
                ));
            }
            let hi = decode_hex_nybble(token.as_bytes()[0] as char)?;
            let lo = decode_hex_nybble(token.as_bytes()[1] as char)?;
            out.push((hi << 4) | lo);
        }
        return Ok(out);
    }

    let normalized = input
        .strip_prefix("0x")
        .or_else(|| input.strip_prefix("0X"))
        .unwrap_or(input)
        .replace('_', "");
    if normalized.is_empty() {
        return Ok(Vec::new());
    }
    if normalized.len() % 2 != 0 {
        return Err("expected an even number of hex digits".to_owned());
    }

    let mut out = Vec::with_capacity(normalized.len() / 2);
    let bytes = normalized.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        let hi = decode_hex_nybble(bytes[index] as char)?;
        let lo = decode_hex_nybble(bytes[index + 1] as char)?;
        out.push((hi << 4) | lo);
        index += 2;
    }
    Ok(out)
}

fn decode_hex_nybble(ch: char) -> Result<u8, String> {
    match ch {
        '0'..='9' => Ok((ch as u8) - b'0'),
        'a'..='f' => Ok((ch as u8) - b'a' + 10),
        'A'..='F' => Ok((ch as u8) - b'A' + 10),
        _ => Err(format!("invalid hex digit `{ch}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_hex_bytes;

    #[test]
    fn parse_hex_bytes_accepts_compact_hex() {
        assert_eq!(parse_hex_bytes("808001").unwrap(), vec![0x80, 0x80, 0x01]);
    }

    #[test]
    fn parse_hex_bytes_accepts_byte_list() {
        assert_eq!(
            parse_hex_bytes("[0x80, 0x80, 0x01]").unwrap(),
            vec![0x80, 0x80, 0x01]
        );
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

    print_mcp_setup_instructions(&dst);
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

fn print_mcp_setup_instructions(installed_binary: &Path) {
    let binary = installed_binary.display();
    println!();
    println!("MCP setup (copy/paste):");
    println!("  codex  => codex mcp add kajit-mir -- {binary}");
    println!("  claude => claude mcp add --transport stdio kajit-mir -- {binary}");
    println!();
    println!("After adding, restart the client so it picks up the new MCP server.");
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent directory");
    }
    fs::write(path, content).expect("write generated file");
}
