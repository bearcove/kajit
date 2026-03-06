use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use kajit::ir::{LambdaId, VReg};
use kajit_mir::DebugCfgMirCommand;
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

#[allow(dead_code)]
struct IrOptCase {
    name: &'static str,
    ir: &'static str,
    must_not_contain_after: &'static [&'static str],
    must_contain_after: &'static [&'static str],
    input: &'static [u8],
}

#[allow(dead_code)]
struct IrPostRegCase {
    name: &'static str,
    ir: &'static str,
    must_contain_linear: &'static [&'static str],
    max_total_edits: usize,
    input: &'static [u8],
    expected: u8,
}

#[allow(dead_code)]
struct BehaviorVector {
    input: &'static [u8],
    expected: u8,
}

#[allow(dead_code)]
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
        Some("corpus-cfg-mir") => print_corpus_cfg_mir(&command_args),
        Some("corpus-input") => print_corpus_input(&command_args),
        Some("minimize-cfg-mir") => minimize_cfg_mir(&command_args),
        Some("debug-cfg-mir") => debug_cfg_mir(&command_args),
        _ => {
            eprintln!(
                "usage: cargo run --manifest-path xtask/Cargo.toml -- <install|gen|corpus-cfg-mir|corpus-input|minimize-cfg-mir|debug-cfg-mir>"
            );
            std::process::exit(2);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DebugCfgMirArgs {
    command: DebugCfgMirCommand,
    path: String,
    input: Vec<u8>,
    input_label: String,
    corpus_test: Option<String>,
}

fn debug_cfg_mir(args: &[String]) {
    let Some(parsed) = parse_debug_cfg_mir_args(args) else {
        eprintln!(
            "usage: cargo run --manifest-path xtask/Cargo.toml -- debug-cfg-mir <run|trace|diff> <cfg-mir-path> <hex-input>\n       cargo run --manifest-path xtask/Cargo.toml -- debug-cfg-mir <run|trace|diff> <cfg-mir-path> --corpus-test <exact-corpus-test-name>\n       cargo run --manifest-path xtask/Cargo.toml -- debug-cfg-mir why-vreg <cfg-mir-path> <hex-input> --vreg <index>\n       cargo run --manifest-path xtask/Cargo.toml -- debug-cfg-mir why-vreg <cfg-mir-path> --corpus-test <exact-corpus-test-name> --vreg <index>\n       cargo run --manifest-path xtask/Cargo.toml -- debug-cfg-mir block <cfg-mir-path> <hex-input> --block <id> [--lambda <id>]\n       cargo run --manifest-path xtask/Cargo.toml -- debug-cfg-mir block <cfg-mir-path> --corpus-test <exact-corpus-test-name> --block <id> [--lambda <id>]"
        );
        std::process::exit(2);
    };

    let output = if let Some(test_name) = parsed.corpus_test.as_deref() {
        debug_cfg_mir_via_corpus_test(&parsed, test_name)
    } else {
        debug_cfg_mir_locally(&parsed)
    }
    .unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    print!("{output}");
}

fn parse_debug_cfg_mir_args(args: &[String]) -> Option<DebugCfgMirArgs> {
    let (command_name, rest) = args.split_first()?;
    let command = match command_name.as_str() {
        "run" => DebugCfgMirCommand::Run,
        "trace" => DebugCfgMirCommand::Trace,
        "diff" => DebugCfgMirCommand::Diff,
        "why-vreg" => DebugCfgMirCommand::WhyVreg { vreg: VReg::new(0) },
        "block" => DebugCfgMirCommand::Block {
            lambda: LambdaId::new(0),
            block: kajit_mir::cfg_mir::BlockId(0),
        },
        _ => return None,
    };

    let (path, rest) = rest.split_first()?;
    let mut input = None::<Vec<u8>>;
    let mut input_label = None::<String>;
    let mut corpus_test = None::<String>;
    let mut vreg = None::<usize>;
    let mut block = None::<u32>;
    let mut lambda = 0usize;
    let mut index = 0usize;
    while index < rest.len() {
        match rest[index].as_str() {
            "--corpus-test" => {
                let test_name = rest.get(index + 1)?.clone();
                let input_hex = fetch_corpus_input_hex(&test_name).ok()?;
                let parsed = parse_hex_bytes(&input_hex).ok()?;
                input = Some(parsed);
                input_label = Some(format!("corpus test {test_name} ({input_hex})"));
                corpus_test = Some(test_name);
                index += 2;
            }
            "--vreg" => {
                vreg = Some(rest.get(index + 1)?.parse().ok()?);
                index += 2;
            }
            "--block" => {
                block = Some(rest.get(index + 1)?.parse().ok()?);
                index += 2;
            }
            "--lambda" => {
                lambda = rest.get(index + 1)?.parse().ok()?;
                index += 2;
            }
            value if !value.starts_with("--") && input.is_none() => {
                let parsed = parse_hex_bytes(value).ok()?;
                input = Some(parsed);
                input_label = Some(value.to_owned());
                index += 1;
            }
            _ => return None,
        }
    }

    let input = input?;
    let input_label = input_label?;
    let command = match command {
        DebugCfgMirCommand::Run => DebugCfgMirCommand::Run,
        DebugCfgMirCommand::Trace => DebugCfgMirCommand::Trace,
        DebugCfgMirCommand::Diff => DebugCfgMirCommand::Diff,
        DebugCfgMirCommand::WhyVreg { .. } => DebugCfgMirCommand::WhyVreg {
            vreg: VReg::new(vreg? as u32),
        },
        DebugCfgMirCommand::Block { .. } => DebugCfgMirCommand::Block {
            lambda: LambdaId::new(lambda as u32),
            block: kajit_mir::cfg_mir::BlockId(block?),
        },
    };

    Some(DebugCfgMirArgs {
        command,
        path: path.clone(),
        input,
        input_label,
        corpus_test,
    })
}

fn debug_cfg_mir_locally(parsed: &DebugCfgMirArgs) -> Result<String, String> {
    let text = fs::read_to_string(&parsed.path)
        .map_err(|err| format!("failed to read {}: {err}", parsed.path))?;
    let registry = kajit::known_intrinsic_registry();
    let program = kajit_mir_text::parse_cfg_mir_with_registry(&text, &registry).map_err(|err| {
        format!(
            "failed to parse CFG-MIR from {}: {err}\nIf this file contains shape-derived symbols, rerun with --corpus-test <exact-corpus-test-name>.",
            parsed.path
        )
    })?;
    run_debug_cfg_mir_command(&program, &parsed.input, &parsed.command)
}

fn debug_cfg_mir_via_corpus_test(
    parsed: &DebugCfgMirArgs,
    test_name: &str,
) -> Result<String, String> {
    let filter = format!("test(={test_name})");
    let root = workspace_root();
    let path = fs::canonicalize(&parsed.path)
        .map_err(|err| format!("failed to canonicalize {}: {err}", parsed.path))?;

    let list_output = Command::new("cargo")
        .args([
            "nextest", "list", "-p", "kajit", "--test", "corpus", "-E", &filter,
        ])
        .current_dir(&root)
        .output()
        .map_err(|err| format!("failed to run cargo nextest list: {err}"))?;
    if !list_output.status.success() {
        let stderr = String::from_utf8_lossy(&list_output.stderr);
        return Err(format!("cargo nextest list failed:\n{stderr}"));
    }

    let mut command = Command::new("cargo");
    command
        .args([
            "nextest",
            "run",
            "-p",
            "kajit",
            "--test",
            "corpus",
            "--run-ignored",
            "all",
            "--no-capture",
            "-E",
            &filter,
        ])
        .env("KAJIT_DEBUG_CFG_MIR", &path)
        .env(
            "KAJIT_DEBUG_CFG_MIR_COMMAND",
            debug_cfg_mir_command_name(&parsed.command),
        )
        .current_dir(&root);
    match &parsed.command {
        DebugCfgMirCommand::WhyVreg { vreg } => {
            command.env("KAJIT_DEBUG_CFG_MIR_VREG", vreg.index().to_string());
        }
        DebugCfgMirCommand::Block { lambda, block } => {
            command.env("KAJIT_DEBUG_CFG_MIR_BLOCK", block.0.to_string());
            command.env("KAJIT_DEBUG_CFG_MIR_LAMBDA", lambda.index().to_string());
        }
        DebugCfgMirCommand::Run | DebugCfgMirCommand::Trace | DebugCfgMirCommand::Diff => {}
    }

    let run_output = command
        .output()
        .map_err(|err| format!("failed to run cargo nextest run: {err}"))?;

    let stdout = String::from_utf8_lossy(&run_output.stdout);
    let stderr = String::from_utf8_lossy(&run_output.stderr);
    let Some(output) = extract_case_debug_cfg_mir_output(&stdout)
        .or_else(|| extract_case_debug_cfg_mir_output(&stderr))
    else {
        let mut message = if !run_output.status.success() {
            format!(
                "cargo nextest run failed before emitting debug CFG-MIR output for `{test_name}`"
            )
        } else {
            format!("debug CFG-MIR output block was not found for `{test_name}`")
        };
        if !stdout.trim().is_empty() {
            message.push_str("\nstdout:\n");
            message.push_str(&stdout);
        }
        if !stderr.trim().is_empty() {
            message.push_str("\nstderr:\n");
            message.push_str(&stderr);
        }
        return Err(message);
    };

    Ok(output)
}

fn debug_cfg_mir_command_name(command: &DebugCfgMirCommand) -> &'static str {
    match command {
        DebugCfgMirCommand::Run => "run",
        DebugCfgMirCommand::Trace => "trace",
        DebugCfgMirCommand::Diff => "diff",
        DebugCfgMirCommand::WhyVreg { .. } => "why-vreg",
        DebugCfgMirCommand::Block { .. } => "block",
    }
}

fn run_debug_cfg_mir_command(
    program: &kajit_mir::cfg_mir::Program,
    input: &[u8],
    command: &DebugCfgMirCommand,
) -> Result<String, String> {
    kajit_mir::run_debug_cfg_mir_command(program, input, command)
}

fn minimize_cfg_mir(args: &[String]) {
    let Some((path, input, input_label, corpus_test)) = parse_minimize_cfg_mir_args(args) else {
        eprintln!(
            "usage: cargo run --manifest-path xtask/Cargo.toml -- minimize-cfg-mir <cfg-mir-path> <hex-input>\n       cargo run --manifest-path xtask/Cargo.toml -- minimize-cfg-mir <cfg-mir-path> --corpus-test <exact-corpus-test-name>"
        );
        std::process::exit(2);
    };

    if let Some(test_name) = corpus_test.as_deref() {
        let reduced = minimize_cfg_mir_via_corpus_test(path, test_name).unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        });
        print!("{reduced}");
        return;
    }

    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        eprintln!("failed to read {}: {err}", path);
        std::process::exit(1);
    });
    let registry = kajit::known_intrinsic_registry();
    let program =
        kajit_mir_text::parse_cfg_mir_with_registry(&text, &registry).unwrap_or_else(|err| {
            eprintln!("failed to parse CFG-MIR from {}: {err}", path);
            std::process::exit(1);
        });
    let (reduced, stats, interestingness) =
        kajit_mir::minimize_cfg_program_for_differential(&program, &input).unwrap_or_else(|err| {
            match err {
                kajit_mir::MinimizeError::NotInteresting => {
                    eprintln!(
                        "seed program is not differentially interesting for input {input_label}"
                    );
                }
                kajit_mir::MinimizeError::Predicate(message) => {
                    eprintln!("differential minimization failed: {message}");
                }
            }
            std::process::exit(1);
        });

    eprintln!(
        "reduced CFG-MIR for input {input_label}: blocks {} -> {}, insts {} -> {}, edges {} -> {}, accepted {} / {}",
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

fn print_corpus_input(args: &[String]) {
    let [test_name] = args else {
        eprintln!(
            "usage: cargo run --manifest-path xtask/Cargo.toml -- corpus-input <exact-corpus-test-name>"
        );
        std::process::exit(2);
    };

    let hex = fetch_corpus_input_hex(test_name).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    println!("{hex}");
}

fn print_corpus_cfg_mir(args: &[String]) {
    let [test_name] = args else {
        eprintln!(
            "usage: cargo run --manifest-path xtask/Cargo.toml -- corpus-cfg-mir <exact-corpus-test-name>"
        );
        std::process::exit(2);
    };

    let cfg_mir = fetch_corpus_cfg_mir_text(test_name).unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });

    print!("{cfg_mir}");
}

fn parse_minimize_cfg_mir_args(args: &[String]) -> Option<(&str, Vec<u8>, String, Option<String>)> {
    match args {
        [path, input_hex] => {
            let input = parse_hex_bytes(input_hex).unwrap_or_else(|err| {
                eprintln!("invalid hex input `{input_hex}`: {err}");
                std::process::exit(2);
            });
            Some((path.as_str(), input, input_hex.clone(), None))
        }
        [path, flag, test_name] if flag == "--corpus-test" => {
            let input_hex = fetch_corpus_input_hex(test_name).unwrap_or_else(|err| {
                eprintln!("{err}");
                std::process::exit(1);
            });
            let input = parse_hex_bytes(&input_hex).unwrap_or_else(|err| {
                eprintln!(
                    "internal error: failed to parse extracted corpus input `{input_hex}`: {err}"
                );
                std::process::exit(1);
            });
            Some((
                path.as_str(),
                input,
                format!("corpus test {test_name} ({input_hex})"),
                Some(test_name.clone()),
            ))
        }
        _ => None,
    }
}

fn minimize_cfg_mir_via_corpus_test(path: &str, test_name: &str) -> Result<String, String> {
    let filter = format!("test(={test_name})");
    let root = workspace_root();
    let path =
        fs::canonicalize(path).map_err(|err| format!("failed to canonicalize {}: {err}", path))?;

    let list_output = Command::new("cargo")
        .args([
            "nextest", "list", "-p", "kajit", "--test", "corpus", "-E", &filter,
        ])
        .current_dir(&root)
        .output()
        .map_err(|err| format!("failed to run cargo nextest list: {err}"))?;
    if !list_output.status.success() {
        let stderr = String::from_utf8_lossy(&list_output.stderr);
        return Err(format!("cargo nextest list failed:\n{stderr}"));
    }

    let run_output = Command::new("cargo")
        .args([
            "nextest",
            "run",
            "-p",
            "kajit",
            "--test",
            "corpus",
            "--run-ignored",
            "all",
            "--no-capture",
            "-E",
            &filter,
        ])
        .env("KAJIT_MINIMIZE_CFG_MIR", &path)
        .current_dir(&root)
        .output()
        .map_err(|err| format!("failed to run cargo nextest run: {err}"))?;

    let stdout = String::from_utf8_lossy(&run_output.stdout);
    let stderr = String::from_utf8_lossy(&run_output.stderr);
    let Some(cfg_mir) =
        extract_case_minimized_cfg_mir(&stdout).or_else(|| extract_case_minimized_cfg_mir(&stderr))
    else {
        if let Some(message) = extract_minimizer_status_line(&stderr)
            .or_else(|| extract_minimizer_status_line(&stdout))
        {
            return Err(message);
        }

        let mut message = if !run_output.status.success() {
            format!("cargo nextest run failed before emitting minimized CFG-MIR for `{test_name}`")
        } else {
            format!("minimized CFG-MIR block was not found for `{test_name}`")
        };
        if !stdout.trim().is_empty() {
            message.push_str("\nstdout:\n");
            message.push_str(&stdout);
        }
        if !stderr.trim().is_empty() {
            message.push_str("\nstderr:\n");
            message.push_str(&stderr);
        }
        return Err(message);
    };

    Ok(cfg_mir)
}

fn fetch_corpus_input_hex(test_name: &str) -> Result<String, String> {
    let filter = format!("test(={test_name})");
    let root = workspace_root();

    let list_output = Command::new("cargo")
        .args([
            "nextest", "list", "-p", "kajit", "--test", "corpus", "-E", &filter,
        ])
        .current_dir(&root)
        .output()
        .map_err(|err| format!("failed to run cargo nextest list: {err}"))?;
    if !list_output.status.success() {
        let stderr = String::from_utf8_lossy(&list_output.stderr);
        return Err(format!("cargo nextest list failed:\n{stderr}"));
    }
    let listed = String::from_utf8_lossy(&list_output.stdout);
    let matches = listed
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        let mut message = format!(
            "expected exactly one corpus test for `{test_name}`, got {}",
            matches.len()
        );
        if !matches.is_empty() {
            message.push_str("\nmatches:");
            for line in matches {
                message.push_str("\n  ");
                message.push_str(line);
            }
        }
        return Err(message);
    }

    let run_output = Command::new("cargo")
        .args([
            "nextest",
            "run",
            "-p",
            "kajit",
            "--test",
            "corpus",
            "--run-ignored",
            "all",
            "--no-capture",
            "-E",
            &filter,
        ])
        .env("KAJIT_PRINT_INPUT_HEX", "1")
        .current_dir(&root)
        .output()
        .map_err(|err| format!("failed to run cargo nextest run: {err}"))?;

    let stdout = String::from_utf8_lossy(&run_output.stdout);
    let stderr = String::from_utf8_lossy(&run_output.stderr);
    let Some(hex) = extract_case_input_hex(&stdout).or_else(|| extract_case_input_hex(&stderr))
    else {
        let mut message = if !run_output.status.success() {
            format!("cargo nextest run failed before emitting case input for `{test_name}`")
        } else {
            format!("case input line was not found for `{test_name}`")
        };
        if !stdout.trim().is_empty() {
            message.push_str("\nstdout:\n");
            message.push_str(&stdout);
        }
        if !stderr.trim().is_empty() {
            message.push_str("\nstderr:\n");
            message.push_str(&stderr);
        }
        return Err(message);
    };

    Ok(hex)
}

fn fetch_corpus_cfg_mir_text(test_name: &str) -> Result<String, String> {
    let filter = format!("test(={test_name})");
    let root = workspace_root();

    let list_output = Command::new("cargo")
        .args([
            "nextest", "list", "-p", "kajit", "--test", "corpus", "-E", &filter,
        ])
        .current_dir(&root)
        .output()
        .map_err(|err| format!("failed to run cargo nextest list: {err}"))?;
    if !list_output.status.success() {
        let stderr = String::from_utf8_lossy(&list_output.stderr);
        return Err(format!("cargo nextest list failed:\n{stderr}"));
    }
    let listed = String::from_utf8_lossy(&list_output.stdout);
    let matches = listed
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        let mut message = format!(
            "expected exactly one corpus test for `{test_name}`, got {}",
            matches.len()
        );
        if !matches.is_empty() {
            message.push_str("\nmatches:");
            for line in matches {
                message.push_str("\n  ");
                message.push_str(line);
            }
        }
        return Err(message);
    }

    let run_output = Command::new("cargo")
        .args([
            "nextest",
            "run",
            "-p",
            "kajit",
            "--test",
            "corpus",
            "--run-ignored",
            "all",
            "--no-capture",
            "-E",
            &filter,
        ])
        .env("KAJIT_PRINT_CFG_MIR", "1")
        .current_dir(&root)
        .output()
        .map_err(|err| format!("failed to run cargo nextest run: {err}"))?;

    let stdout = String::from_utf8_lossy(&run_output.stdout);
    let stderr = String::from_utf8_lossy(&run_output.stderr);
    let Some(cfg_mir) = extract_case_cfg_mir(&stdout).or_else(|| extract_case_cfg_mir(&stderr))
    else {
        let mut message = if !run_output.status.success() {
            format!("cargo nextest run failed before emitting CFG-MIR for `{test_name}`")
        } else {
            format!("CFG-MIR block was not found for `{test_name}`")
        };
        if !stdout.trim().is_empty() {
            message.push_str("\nstdout:\n");
            message.push_str(&stdout);
        }
        if !stderr.trim().is_empty() {
            message.push_str("\nstderr:\n");
            message.push_str(&stderr);
        }
        return Err(message);
    };

    Ok(cfg_mir)
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

fn extract_case_input_hex(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        line.trim()
            .strip_prefix("KAJIT_CASE_INPUT_HEX=")
            .map(str::to_owned)
    })
}

fn extract_case_cfg_mir(output: &str) -> Option<String> {
    let begin = "KAJIT_CASE_CFG_MIR_BEGIN";
    let end = "KAJIT_CASE_CFG_MIR_END";
    let start = output.find(begin)?;
    let rest = &output[start + begin.len()..];
    let end_index = rest.find(end)?;
    Some(rest[..end_index].trim_matches('\n').to_owned() + "\n")
}

fn extract_case_minimized_cfg_mir(output: &str) -> Option<String> {
    let begin = "KAJIT_CASE_MINIMIZED_CFG_MIR_BEGIN";
    let end = "KAJIT_CASE_MINIMIZED_CFG_MIR_END";
    let start = output.find(begin)?;
    let rest = &output[start + begin.len()..];
    let end_index = rest.find(end)?;
    Some(rest[..end_index].trim_matches('\n').to_owned() + "\n")
}

fn extract_case_debug_cfg_mir_output(output: &str) -> Option<String> {
    let begin = "KAJIT_CASE_DEBUG_CFG_MIR_BEGIN";
    let end = "KAJIT_CASE_DEBUG_CFG_MIR_END";
    let start = output.find(begin)?;
    let rest = &output[start + begin.len()..];
    let end_index = rest.find(end)?;
    Some(rest[..end_index].trim_matches('\n').to_owned() + "\n")
}

fn extract_minimizer_status_line(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let line = line.trim();
        if line.starts_with("seed program is not differentially interesting")
            || line.starts_with("differential minimization failed:")
        {
            Some(line.to_owned())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use kajit::ir::{LambdaId, VReg};

    use super::{
        DebugCfgMirCommand, extract_case_cfg_mir, extract_case_debug_cfg_mir_output,
        extract_case_input_hex, extract_case_minimized_cfg_mir, extract_minimizer_status_line,
        parse_debug_cfg_mir_args, parse_hex_bytes, parse_minimize_cfg_mir_args,
    };

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

    #[test]
    fn extract_case_input_hex_finds_structured_line() {
        assert_eq!(
            extract_case_input_hex("noise\nKAJIT_CASE_INPUT_HEX=2a00ff\nmore noise"),
            Some("2a00ff".to_owned())
        );
    }

    #[test]
    fn extract_case_cfg_mir_finds_marker_block() {
        assert_eq!(
            extract_case_cfg_mir(
                "noise\nKAJIT_CASE_CFG_MIR_BEGIN\nfunc @0 {}\nKAJIT_CASE_CFG_MIR_END\nmore"
            ),
            Some("func @0 {}\n".to_owned())
        );
    }

    #[test]
    fn parse_minimize_cfg_mir_args_accepts_raw_hex() {
        let args = vec!["seed.cfg".to_owned(), "2a00ff".to_owned()];
        let (path, input, label, corpus_test) =
            parse_minimize_cfg_mir_args(&args).expect("args should parse");
        assert_eq!(path, "seed.cfg");
        assert_eq!(input, vec![0x2a, 0x00, 0xff]);
        assert_eq!(label, "2a00ff");
        assert_eq!(corpus_test, None);
    }

    #[test]
    fn extract_case_minimized_cfg_mir_finds_marker_block() {
        assert_eq!(
            extract_case_minimized_cfg_mir(
                "noise\nKAJIT_CASE_MINIMIZED_CFG_MIR_BEGIN\nfunc @0 {}\nKAJIT_CASE_MINIMIZED_CFG_MIR_END\nmore"
            ),
            Some("func @0 {}\n".to_owned())
        );
    }

    #[test]
    fn extract_case_debug_cfg_mir_output_finds_marker_block() {
        assert_eq!(
            extract_case_debug_cfg_mir_output(
                "noise\nKAJIT_CASE_DEBUG_CFG_MIR_BEGIN\ncursor: 1\nKAJIT_CASE_DEBUG_CFG_MIR_END\nmore"
            ),
            Some("cursor: 1\n".to_owned())
        );
    }

    #[test]
    fn parse_debug_cfg_mir_args_accepts_why_vreg() {
        let args = vec![
            "why-vreg".to_owned(),
            "seed.cfg".to_owned(),
            "2a".to_owned(),
            "--vreg".to_owned(),
            "47".to_owned(),
        ];
        let parsed = parse_debug_cfg_mir_args(&args).expect("args should parse");
        assert_eq!(parsed.path, "seed.cfg");
        assert_eq!(parsed.input, vec![0x2a]);
        assert_eq!(parsed.input_label, "2a");
        assert_eq!(parsed.corpus_test, None);
        assert_eq!(
            parsed.command,
            DebugCfgMirCommand::WhyVreg {
                vreg: VReg::new(47)
            }
        );
    }

    #[test]
    fn parse_debug_cfg_mir_args_accepts_block_query() {
        let args = vec![
            "block".to_owned(),
            "seed.cfg".to_owned(),
            "2a".to_owned(),
            "--block".to_owned(),
            "17".to_owned(),
            "--lambda".to_owned(),
            "1".to_owned(),
        ];
        let parsed = parse_debug_cfg_mir_args(&args).expect("args should parse");
        assert_eq!(
            parsed.command,
            DebugCfgMirCommand::Block {
                lambda: LambdaId::new(1),
                block: kajit_mir::cfg_mir::BlockId(17)
            }
        );
    }

    #[test]
    fn extract_minimizer_status_line_finds_interestingness_failure() {
        assert_eq!(
            extract_minimizer_status_line(
                "noise\nseed program is not differentially interesting for input 012a\nmore"
            ),
            Some("seed program is not differentially interesting for input 012a".to_owned())
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
