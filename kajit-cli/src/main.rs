mod mcp;

use facet::Facet;
use figue as args;

/// kajit — JIT deserializer toolkit
#[derive(Facet, Debug)]
struct Args {
    #[facet(args::subcommand)]
    command: Command,
}

#[derive(Facet, Debug)]
#[repr(u8)]
enum Command {
    /// Run the MCP debugger server
    Mcp {
        /// Run in real mode (direct MCP protocol, not proxy)
        #[facet(args::named, default)]
        real: bool,
    },

    /// Evaluate a CFG-MIR program with the ideal interpreter
    Eval {
        /// Path to CFG-MIR text file
        #[facet(args::positional)]
        cfg_mir: String,

        /// Input bytes as hex string
        #[facet(args::positional, default)]
        input_hex: Option<String>,
    },

    /// Compile a type and dump pipeline stages
    Compile {
        /// Format: postcard or json
        #[facet(args::positional)]
        format: String,

        /// Type to compile (e.g. u32, Vec<u8>, MyStruct)
        #[facet(args::positional)]
        ty: String,

        /// Stages to dump: hir, ir, linear, cfg, emit, asm, all
        #[facet(args::named, args::short = 's', default)]
        stage: Option<String>,
    },
}

fn main() {
    let args: Args = figue::from_std_args().unwrap();

    match args.command {
        Command::Mcp { real } => {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            let result = if real {
                rt.block_on(mcp::run_real())
            } else {
                rt.block_on(mcp::run_mcp_proxy())
            };
            if let Err(error) = result {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        Command::Eval { cfg_mir, input_hex } => {
            cmd_eval(&cfg_mir, input_hex.as_deref().unwrap_or(""));
        }
        Command::Compile { format, ty, stage } => {
            cmd_compile(&format, &ty, stage.as_deref().unwrap_or("all"));
        }
    }
}

// ─── eval: ideal interpreter on CFG-MIR ──────────────────────────────────────

fn cmd_eval(cfg_mir_path: &str, input_hex: &str) {
    let mir_text = std::fs::read_to_string(cfg_mir_path).unwrap_or_else(|e| {
        eprintln!("error: failed to read {cfg_mir_path}: {e}");
        std::process::exit(1);
    });

    let program = kajit_mir_text::parse_cfg_mir(&mir_text).unwrap_or_else(|e| {
        eprintln!("error: failed to parse CFG-MIR: {e}");
        std::process::exit(1);
    });

    let input = parse_hex(input_hex);
    let mut session = kajit_mir::DebuggerSession::new(&program, &input).unwrap_or_else(|e| {
        eprintln!("error: failed to create session: {e}");
        std::process::exit(1);
    });

    let _events = session
        .run_until(kajit_mir::RunUntilTarget::Return, 100_000)
        .unwrap_or_else(|e| {
            eprintln!("error: execution failed: {e}");
            std::process::exit(1);
        });

    let state = session.state();
    print_interpreter_state(&state);
}

fn print_interpreter_state(state: &kajit_mir::DebuggerState) {
    println!("steps: {}", state.step_count);
    println!("cursor: {}", state.cursor);
    if let Some(trap) = &state.trap {
        println!("TRAP: {:?} at offset {}", trap.code, trap.offset);
    }
    if state.returned {
        println!("returned: yes");
    }
    println!("output: {}", encode_hex(&state.output));

    let nonzero: Vec<_> = state
        .vregs
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0)
        .collect();
    if !nonzero.is_empty() {
        print!("vregs:");
        for (idx, val) in &nonzero {
            print!(" v{}={}", idx, val);
        }
        println!();
    }
}

// ─── compile: full pipeline dump ─────────────────────────────────────────────

fn cmd_compile(format: &str, ty: &str, stages: &str) {
    let kind = match format {
        "postcard" => kajit::DecoderKind::Postcard,
        "json" => kajit::DecoderKind::Json,
        other => {
            eprintln!("error: unknown format '{other}', expected 'postcard' or 'json'");
            std::process::exit(1);
        }
    };

    let shape = resolve_shape(ty);
    let pipeline_opts = kajit::PipelineOptions::from_env();

    // Single compilation pass — all artifacts share the same vreg numbering
    let artifacts = kajit::compile_pipeline(shape, kind, &pipeline_opts);

    let dump_all = stages == "all";
    let dump = |name: &str| dump_all || stages.split(',').any(|s| s.trim() == name);

    if dump("hir") {
        println!("=== HIR ===");
        println!("{}", artifacts.hir_text);
    }

    if dump("ir") || dump("opts") {
        for (pass_name, ir_text) in &artifacts.ir_opt_timeline {
            let node_count = ir_text.matches(" = ").count();
            println!("=== IR after {pass_name} ({node_count} nodes) ===");
            if dump("ir") {
                println!("{ir_text}");
            }
        }
    }

    if dump("linear") {
        println!("=== Linear IR ===");
        println!("{}", artifacts.linear_text);
    }

    if dump("cfg") {
        let block_count = artifacts.cfg_text.matches("block b").count();
        let inst_count = artifacts.cfg_text.matches("inst i").count();
        let edge_count = artifacts.cfg_text.matches("edge e").count();
        println!("=== CFG-MIR ({block_count} blocks, {inst_count} insts, {edge_count} edges) ===");
        println!("{}", artifacts.cfg_text);
    }

    if dump("asm") || dump("emit") {
        if artifacts.asm_text.is_empty() {
            println!("=== Assembly (not available on this platform) ===");
        } else {
            let inst_count = artifacts
                .asm_text
                .lines()
                .filter(|l| !l.is_empty() && !l.starts_with('.'))
                .count();
            println!("=== Assembly ({inst_count} instructions) ===");
            println!("{}", artifacts.asm_text);
        }
    }

    if dump_all {
        let last_ir = artifacts
            .ir_opt_timeline
            .last()
            .map(|(_, t)| t.as_str())
            .unwrap_or("");
        let ir_nodes = last_ir.matches(" = ").count();
        let cfg_blocks = artifacts.cfg_text.matches("block b").count();
        let cfg_insts = artifacts.cfg_text.matches("inst i").count();
        let cfg_edges = artifacts.cfg_text.matches("edge e").count();
        let asm_insts = artifacts
            .asm_text
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('.'))
            .count();
        println!("=== Stats ===");
        println!("  IR nodes:    {ir_nodes}");
        println!("  CFG blocks:  {cfg_blocks}");
        println!("  CFG insts:   {cfg_insts}");
        println!("  CFG edges:   {cfg_edges}");
        println!("  ASM insts:   {asm_insts}");
    }
}

/// Resolve a type name to a facet Shape.
///
/// Supports built-in types: u8, u16, u32, u64, i8, i16, i32, i64, bool, String
fn resolve_shape(ty: &str) -> &'static facet::Shape {
    use facet::Facet;
    match ty {
        "u8" => u8::SHAPE,
        "u16" => u16::SHAPE,
        "u32" => u32::SHAPE,
        "u64" => u64::SHAPE,
        "i8" => i8::SHAPE,
        "i16" => i16::SHAPE,
        "i32" => i32::SHAPE,
        "i64" => i64::SHAPE,
        "bool" => bool::SHAPE,
        "String" | "string" => String::SHAPE,
        other => {
            eprintln!("error: unknown type '{other}'");
            eprintln!("supported: u8 u16 u32 u64 i8 i16 i32 i64 bool String");
            std::process::exit(1);
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn parse_hex(s: &str) -> Vec<u8> {
    let cleaned: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    if cleaned.is_empty() {
        return Vec::new();
    }
    cleaned
        .as_bytes()
        .chunks_exact(2)
        .map(|chunk| {
            let s = std::str::from_utf8(chunk).unwrap();
            u8::from_str_radix(s, 16).unwrap()
        })
        .collect()
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
