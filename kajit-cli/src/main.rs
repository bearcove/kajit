#[cfg(feature = "lldb")]
mod lldb_debugger;
mod mcp;

use facet::Facet;
use figue as args;

/// kajit — JIT deserializer toolkit
#[derive(Facet, Debug)]
struct Args {
    /// Standard CLI options
    #[facet(flatten)]
    builtins: args::FigueBuiltins,

    #[facet(args::subcommand)]
    command: Command,
}

#[derive(Facet, Debug)]
#[repr(u8)]
enum Command {
    /// Run the Kajit MCP server
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

    /// Compile a source file (.vixen-hir, .vixen-ir, .vixen-mir, .vixen-asm)
    Compile {
        /// Path to source file
        #[facet(args::positional)]
        path: String,

        /// Stages to dump: hir, ir, linear, cfg, emit, asm, all
        #[facet(args::named, args::short = 's', default)]
        stage: Option<String>,
    },

    /// Compile a format decoder for a type and dump pipeline stages
    CompileFormat {
        /// Format: postcard
        #[facet(args::positional)]
        format: String,

        /// Type to compile (e.g. u32, Vec<u8>, MyStruct)
        #[facet(args::positional)]
        ty: String,

        /// Stages to dump: hir, ir, linear, cfg, emit, asm, exec, all
        #[facet(args::named, args::short = 's', default)]
        stage: Option<String>,

        /// Input bytes as hex string (for exec stage)
        #[facet(args::named, args::short = 'i', default)]
        input: Option<String>,

        /// Reduce: find minimal CFG-MIR that triggers divergence or SSA breakage
        #[facet(args::named, default)]
        reduce: Option<String>,

        /// Reduce IR: find minimal RVSDG that satisfies a predicate after passes.
        /// Format: "passes:predicate" e.g. "unroll_const_fold:has_op(Mul)"
        #[facet(args::named, default)]
        reduce_ir: Option<String>,
    },

    /// Minimize a .vixen-hir file while preserving a compilation failure (e.g. SSA violation)
    ReduceHir {
        /// Path to .vixen-hir file
        #[facet(args::positional)]
        path: String,
    },

    /// Lockstep differential debugger: step interpreter + LLDB in parallel
    DebugDiff {
        /// Format: postcard
        #[facet(args::positional)]
        format: String,

        /// Type to compile (e.g. u32, Vec<u8>, MyStruct)
        #[facet(args::positional)]
        ty: String,

        /// Input bytes as hex string
        #[facet(args::positional)]
        input_hex: String,
    },
}

fn main() {
    let args: Args = figue::from_std_args().unwrap();

    // MCP --real mode logs to a file (stdout/stderr are used by MCP protocol).
    // All other commands log to stderr.
    if matches!(args.command, Command::Mcp { real: true }) {
        if let Ok(log_file) = std::fs::File::create("/tmp/kajit-mcp.log") {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
                )
                .with_writer(log_file)
                .with_ansi(false)
                .init();
        }
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .init();
    }

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
        Command::Compile { path, stage } => {
            cmd_compile_file(&path, stage.as_deref().unwrap_or("all"));
        }
        Command::ReduceHir { path } => {
            cmd_reduce_hir(&path);
        }
        Command::CompileFormat {
            format,
            ty,
            stage,
            input,
            reduce,
            reduce_ir,
        } => {
            if let Some(pass_name) = reduce {
                cmd_compile_reduce(&format, &ty, &pass_name);
            } else if let Some(spec) = reduce_ir {
                cmd_reduce_ir(&format, &ty, &spec);
            } else {
                cmd_compile(
                    &format,
                    &ty,
                    stage.as_deref().unwrap_or("all"),
                    input.as_deref(),
                );
            }
        }
        Command::DebugDiff {
            format,
            ty,
            input_hex,
        } => {
            cmd_debug_diff(&format, &ty, &input_hex);
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
    let args = kajit_types::Arguments::new();
    let mut session =
        kajit_mir::DebuggerSession::new(&program, &input, &args).unwrap_or_else(|e| {
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

// ─── compile: source file compilation ───────────────────────────────────────

fn cmd_compile_file(path: &str, stages: &str) {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let source = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error: failed to read {path}: {e}");
        std::process::exit(1);
    });

    match ext {
        "vixen-hir" => {
            let module = kajit_hir_text::parse_hir(&source).unwrap_or_else(|e| {
                eprintln!("error: failed to parse HIR: {e}");
                std::process::exit(1);
            });
            let registry = kajit_ir::IntrinsicRegistry::empty();
            let pipeline_opts = kajit::PipelineOptions::builder()
                .from_env()
                .compile_target(kajit::CompileTarget::Object)
                .build();

            let artifacts =
                kajit::compile_pipeline_from_hir_module(&module, &registry, &pipeline_opts);

            let dump = |name: &str| stages == "all" || stages.split(',').any(|s| s == name);

            if dump("hir") {
                println!("=== HIR ===");
                println!("{}", module);
            }
            if dump("ir") {
                for (label, text) in &artifacts.ir_opt_timeline {
                    println!("=== IR ({label}) ===");
                    println!("{text}");
                }
            }
            if dump("cfg") {
                println!("=== CFG-MIR ===");
                println!("{}", artifacts.cfg_text);
            }
            if dump("asm") || dump("emit") {
                println!("=== ASM ===");
                println!("{}", artifacts.asm_text);
            }

            // Emit object file
            let stem = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let obj_path = format!("{stem}.o");

            let code = artifacts.decoder.code();
            let harness_input = kajit::harness::HarnessInput {
                code,
                entry_offset: artifacts.decoder.entry_offset(),
                output_size: 0,
                dwarf: None,
                cfg_mir_lines: &[],
                function_name: stem,
                alloc_map: None,
                intrinsic_calls: vec![],
                extern_addr_relocs: artifacts.extern_addr_relocs.clone(),
            };
            kajit::harness::build_object_file(&harness_input, std::path::Path::new(&obj_path))
                .unwrap_or_else(|e| {
                    eprintln!("error: failed to write object file: {e}");
                    std::process::exit(1);
                });
            eprintln!("wrote {obj_path}");
        }
        "vixen-ir" | "vixen-mir" | "vixen-asm" => {
            eprintln!("error: .{ext} compilation not yet implemented");
            std::process::exit(1);
        }
        _ => {
            eprintln!(
                "error: unknown file extension '.{ext}', expected one of: \
                 .vixen-hir, .vixen-ir, .vixen-mir, .vixen-asm"
            );
            std::process::exit(1);
        }
    }
}

// ─── compile-format: format decoder pipeline dump ───────────────────────────

fn cmd_compile(format: &str, ty: &str, stages: &str, input_hex: Option<&str>) {
    let kind = match format {
        "postcard" => kajit::DecoderKind::Postcard,
        other => {
            eprintln!("error: unknown format '{other}', expected 'postcard'");
            std::process::exit(1);
        }
    };

    let shape = resolve_shape(ty);
    let pipeline_opts = kajit::PipelineOptions::from_env();

    let dump_all = stages == "all";
    let dump = |name: &str| dump_all || stages.split(',').any(|s| s.trim() == name);

    // Dump HIR early — before the full pipeline which may panic
    if dump("hir") {
        let hir_text = kajit::debug_hir_text(shape, kind);
        println!("=== HIR ===");
        println!("{hir_text}");
        if stages == "hir" {
            return;
        }
    }

    // Single compilation pass — all artifacts share the same vreg numbering
    let artifacts = kajit::compile_pipeline(shape, kind, &pipeline_opts);

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

    if dump("harness") {
        let output_size = shape.layout.sized_layout().map(|l| l.size()).unwrap_or(0);
        let output_dir = std::path::PathBuf::from("/tmp/kajit-harness".to_string());
        let base_name = format!("harness_{format}_{ty}");
        let listing_path = output_dir.join(format!("{base_name}.cfg-mir"));

        let dwarf = artifacts.decoder.build_standalone_dwarf(&listing_path);

        // Map backend call sites to harness call sites (resolve symbol names)
        let known = kajit::intrinsics::known_intrinsics();
        let intrinsic_calls: Vec<_> = artifacts
            .intrinsic_call_sites
            .iter()
            .filter_map(|site| {
                let name = known
                    .iter()
                    .find(|(_, f)| f.0 == site.func.0)
                    .map(|(name, _)| name.to_string())?;
                Some(kajit::harness::IntrinsicCallSite {
                    code_offset: site.code_offset,
                    baked_addr: site.func.0 as u64,
                    symbol_name: name,
                })
            })
            .collect();

        let harness_input = kajit::harness::HarnessInput {
            code: artifacts.decoder.code(),
            entry_offset: artifacts.decoder.entry_offset(),
            output_size,
            dwarf,
            cfg_mir_lines: artifacts.decoder.cfg_mir_lines(),
            function_name: "kajit_decode",
            alloc_map: Some(&artifacts.alloc_map),
            intrinsic_calls,
            extern_addr_relocs: artifacts.extern_addr_relocs.clone(),
        };

        match kajit::harness::generate_harness(&harness_input, &output_dir, &base_name) {
            Ok(exe_path) => {
                println!("=== Harness ===");
                println!("  executable: {}", exe_path.display());
                println!("  listing:    {}", listing_path.display());

                // If we have input, run it
                let input = input_hex
                    .map(parse_hex)
                    .unwrap_or_else(|| make_test_input(format, ty));
                let hex_input: String = input.iter().map(|b| format!("{b:02x}")).collect();

                let result = std::process::Command::new(&exe_path)
                    .arg(&hex_input)
                    .output();
                match result {
                    Ok(output) if output.status.success() => {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        println!("  run({hex_input}): {}", stdout.trim());
                    }
                    Ok(output) => {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        println!("  run({hex_input}): FAILED — {}", stderr.trim());
                    }
                    Err(e) => println!("  run: could not execute — {e}"),
                }
            }
            Err(e) => {
                eprintln!("error generating harness: {e}");
                std::process::exit(1);
            }
        }
    }

    if dump("exec") || dump_all {
        // Get input: either from --input flag or auto-generate
        let input = input_hex
            .map(parse_hex)
            .unwrap_or_else(|| make_test_input(format, ty));

        let output_size = shape.layout.sized_layout().map(|l| l.size()).unwrap_or(0);

        println!("=== Exec ===");
        println!("  input:       {} ({})", encode_hex(&input), input.len());
        println!("  output_size: {output_size}");

        // Run RVSDG interpreter (pre-linearization, ideal semantics).
        eprint!("  [1/3] rvsdg interpreter... ");
        let ir_text = &artifacts.ir_opt_timeline[0].1;
        let ir_registry = kajit::symbol_registry_for_shape(shape);
        match kajit_ir_text::parse_ir(ir_text, &ir_registry) {
            Ok(ir_func) => {
                let input_clone = input.clone();
                let handle = std::thread::spawn(move || {
                    let mut cursor = kajit::context::RuntimeCursor::new(&input_clone);
                    let mut out_buf = vec![0u8; output_size];
                    let mut ctx = kajit::context::DeserContext::from_bytes(&input_clone);
                    let mut args = kajit_types::Arguments::new();
                    args.push_ptr(&mut cursor);
                    args.push_ptr(out_buf.as_mut_ptr());
                    args.push_ptr(&mut ctx);
                    let outcome = kajit_ir::interpret::interpret(
                        &ir_func,
                        kajit_types::SymbolTable::new(),
                        &args,
                    );
                    (outcome, out_buf)
                });
                let timeout = std::time::Duration::from_secs(5);
                let start = std::time::Instant::now();
                loop {
                    if handle.is_finished() {
                        match handle.join() {
                            Ok((outcome, out_buf)) => match &outcome.trap {
                                None => {
                                    eprintln!("ok");
                                    println!(
                                        "  rvsdg out:   {} ({})",
                                        encode_hex(&out_buf),
                                        out_buf.len()
                                    );
                                }
                                Some(t) => {
                                    eprintln!("trap");
                                    println!("  rvsdg:       TRAP ({:?})", t.code);
                                }
                            },
                            Err(_) => {
                                eprintln!("CRASHED");
                                println!("  rvsdg:       interpreter panicked");
                            }
                        }
                        break;
                    }
                    if start.elapsed() > timeout {
                        eprintln!("TIMEOUT ({}s)", timeout.as_secs());
                        println!(
                            "  rvsdg:       timed out (RVSDG interpreter cannot interpret pointer-based ops without real addresses)"
                        );
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
            Err(e) => {
                eprintln!("parse error");
                println!("  rvsdg:       parse error: {e}");
            }
        }

        // Run CFG-MIR interpreter (post-linearization)
        eprint!("  [2/3] cfg-mir interpreter... ");
        let mut interp_cursor = kajit::context::RuntimeCursor::new(&input);
        let mut interp_out = vec![0u8; output_size];
        let mut interp_ctx = kajit::context::DeserContext::from_bytes(&input);
        let mut interp_args = kajit_types::Arguments::new();
        interp_args.push_ptr(&mut interp_cursor);
        interp_args.push(kajit_types::ArgValue::U64(interp_out.as_mut_ptr() as u64));
        interp_args.push_ptr(&mut interp_ctx);
        let interp_result =
            kajit_mir::opt::reduce::interpret(&artifacts.cfg_program, &input, &interp_args);
        match &interp_result {
            kajit_mir::opt::reduce::InterpOutcome::Ok => {
                eprintln!("ok");
                println!(
                    "  interp out:  {} ({})",
                    encode_hex(&interp_out),
                    interp_out.len()
                );
            }
            kajit_mir::opt::reduce::InterpOutcome::Trapped(trap) => {
                eprintln!("trap");
                println!(
                    "  interp:      TRAP ({:?} at offset {})",
                    trap.code, trap.offset
                );
            }
            kajit_mir::opt::reduce::InterpOutcome::TimedOut => {
                eprintln!("TIMEOUT");
                println!("  interp:      TIMEOUT");
            }
        }

        // Run JIT (may hang or crash on buggy programs)
        eprint!("  [3/3] jit... ");
        let decoder_ptr = &artifacts.decoder as *const kajit::CompiledDecoder as usize;
        let input_clone = input.clone();
        let jit_handle = std::thread::spawn(move || {
            let decoder = unsafe { &*(decoder_ptr as *const kajit::CompiledDecoder) };
            kajit::deserialize_raw(decoder, &input_clone, output_size)
        });
        let timeout = std::time::Duration::from_secs(5);
        let start = std::time::Instant::now();
        let jit_result: Option<Result<Vec<u8>, kajit::DeserError>> = loop {
            if jit_handle.is_finished() {
                match jit_handle.join() {
                    Ok(Ok(bytes)) => {
                        eprintln!("ok");
                        println!("  jit output:  {} ({})", encode_hex(&bytes), bytes.len());
                        break Some(Ok(bytes));
                    }
                    Ok(Err(e)) => {
                        eprintln!("error");
                        println!("  jit error:   {e}");
                        break Some(Err(e));
                    }
                    Err(_) => {
                        eprintln!("CRASHED");
                        println!("  jit:         panicked or crashed");
                        break None;
                    }
                }
            }
            if start.elapsed() > timeout {
                eprintln!("TIMEOUT ({}s)", timeout.as_secs());
                println!("  jit:         timed out");
                break None;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        };

        // Compare
        use kajit_mir::opt::reduce::InterpOutcome;
        match (&jit_result, &interp_result) {
            (Some(Ok(jit)), InterpOutcome::Ok) if *jit == interp_out => {
                println!("  match:       YES");
            }
            (Some(Ok(jit)), InterpOutcome::Ok) => {
                println!("  match:       NO — DIVERGENCE");
                let max_len = jit.len().max(interp_out.len());
                for i in 0..max_len {
                    let j = jit.get(i).copied();
                    let r = interp_out.get(i).copied();
                    if j != r {
                        println!(
                            "    byte[{i}]: jit={} interp={}",
                            j.map(|b| format!("0x{b:02x}"))
                                .unwrap_or_else(|| "---".to_string()),
                            r.map(|b| format!("0x{b:02x}"))
                                .unwrap_or_else(|| "---".to_string()),
                        );
                    }
                }
            }
            _ => {
                println!("  match:       N/A (one side errored)");
            }
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
fn try_resolve_shape(ty: &str) -> Result<&'static facet::Shape, String> {
    use facet::Facet;
    match ty {
        "u8" => Ok(u8::SHAPE),
        "u16" => Ok(u16::SHAPE),
        "u32" => Ok(u32::SHAPE),
        "u64" => Ok(u64::SHAPE),
        "i8" => Ok(i8::SHAPE),
        "i16" => Ok(i16::SHAPE),
        "i32" => Ok(i32::SHAPE),
        "i64" => Ok(i64::SHAPE),
        "bool" => Ok(bool::SHAPE),
        "String" | "string" => Ok(String::SHAPE),
        "Vec<u8>" => Ok(Vec::<u8>::SHAPE),
        "Vec<u32>" => Ok(Vec::<u32>::SHAPE),
        "Vec<u64>" => Ok(Vec::<u64>::SHAPE),
        "Vec<String>" => Ok(Vec::<String>::SHAPE),
        "Option<u32>" => Ok(Option::<u32>::SHAPE),
        "Option<String>" => Ok(Option::<String>::SHAPE),
        "(u32, u32)" => Ok(<(u32, u32)>::SHAPE),
        other => Err(format!(
            "unknown type '{other}'. Supported: u8 u16 u32 u64 i8 i16 i32 i64 bool String \
             Vec<u8> Vec<u32> Vec<u64> Vec<String> Option<u32> Option<String> (u32, u32)"
        )),
    }
}

fn resolve_shape(ty: &str) -> &'static facet::Shape {
    match try_resolve_shape(ty) {
        Ok(shape) => shape,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

// ─── compile --reduce-ir: minimize RVSDG reproducer ──────────────────────────

fn cmd_reduce_ir(format: &str, ty: &str, spec: &str) {
    // Parse spec: "passes:predicate" e.g. "unroll_const_fold:has_op(Mul)"
    let (passes_str, predicate_str) = spec.split_once(':').unwrap_or_else(|| {
        eprintln!("error: --reduce-ir format is 'passes:predicate'");
        eprintln!("  passes: comma-separated pass names (or 'none')");
        eprintln!("  predicates:");
        eprintln!("    has_op(OpName) — IR contains a node with this op after passes");
        eprintln!("    node_count_gt(N) — more than N nodes after passes");
        eprintln!("  example: --reduce-ir 'unroll_const_fold:has_op(Mul)'");
        std::process::exit(1);
    });

    let kind = match format {
        "postcard" => kajit::DecoderKind::Postcard,
        other => {
            eprintln!("error: unknown format '{other}'");
            std::process::exit(1);
        }
    };

    let shape = resolve_shape(ty);
    let registry = std::rc::Rc::new(kajit::symbol_registry_for_shape(shape));

    // Build initial IR by running the pipeline and extracting the initial IR text.
    let pipeline_opts = kajit::PipelineOptions::from_env();
    let artifacts = kajit::compile_pipeline(shape, kind, &pipeline_opts);

    // Parse the initial (pre-optimization) IR from the timeline.
    let initial_ir_text = &artifacts.ir_opt_timeline[0].1;
    let func = kajit_ir_text::parse_ir(initial_ir_text, &registry).unwrap_or_else(|e| {
        eprintln!("error: failed to parse initial IR: {e}");
        std::process::exit(1);
    });

    eprintln!("[reduce-ir] initial: {} nodes", func.nodes.iter().count());

    // Build the pass runner.
    let pass_names: Vec<&str> = passes_str.split(',').map(|s| s.trim()).collect();
    let run_passes = move |func: &mut kajit_ir::IrFunc| {
        for pass_name in &pass_names {
            match *pass_name {
                "none" => {}
                "unroll" => {
                    kajit_ir::unroll_theta::unroll_bounded_thetas(func);
                }
                "const_fold" => {
                    kajit_ir::const_fold::const_fold(func);
                }
                "simplify_gammas" => {
                    kajit_ir::simplify_gamma::simplify_trivial_gammas(func);
                }
                "unroll_const_fold" => {
                    kajit_ir::unroll_theta::unroll_bounded_thetas(func);
                    kajit_ir::const_fold::const_fold(func);
                    kajit_ir::simplify_gamma::simplify_trivial_gammas(func);
                }
                "all" => {
                    for pass in kajit_ir::default_pass_registry() {
                        pass.run(func);
                    }
                }
                other => {
                    eprintln!("warning: unknown pass '{other}', skipping");
                }
            }
        }
    };

    // Build the predicate.
    let predicate_str = predicate_str.to_string();
    let reg1 = registry.clone();
    let is_interesting = move |func: &kajit_ir::IrFunc| -> bool {
        // Clone via text round-trip for the pass run (don't mutate the candidate).
        let text = format!("{}", func.display_with_registry(&reg1));
        let mut test_func = match kajit_ir_text::parse_ir(&text, &reg1) {
            Ok(f) => f,
            Err(_) => return false,
        };

        run_passes(&mut test_func);

        // Check predicate.
        if let Some(op_name) = predicate_str
            .strip_prefix("has_op(")
            .and_then(|s| s.strip_suffix(')'))
        {
            // Check if any node has this op name.
            test_func.nodes.iter().any(|(_, node)| {
                let kind_str = format!("{:?}", node.kind);
                kind_str.contains(op_name)
            })
        } else if let Some(n_str) = predicate_str
            .strip_prefix("node_count_gt(")
            .and_then(|s| s.strip_suffix(')'))
        {
            let n: usize = n_str.parse().unwrap_or(0);
            test_func.nodes.iter().count() > n
        } else {
            eprintln!("error: unknown predicate '{predicate_str}'");
            false
        }
    };

    // Compact via text round-trip to flush orphaned arena entries between rounds.
    let reg3 = registry.clone();
    let compact_fn = move |func: &kajit_ir::IrFunc| -> kajit_ir::IrFunc {
        let text = format!("{}", func.display_with_registry(&reg3));
        kajit_ir_text::parse_ir(&text, &reg3).expect("compact round-trip failed")
    };

    let (reduced, stats) = kajit_ir::reduce::reduce_ir(&func, &is_interesting, Some(&compact_fn));

    eprintln!(
        "[reduce-ir] done: {} → {} nodes ({} candidates, {} reductions)",
        stats.initial_nodes, stats.final_nodes, stats.candidates_tested, stats.reductions_applied
    );

    // Output the reduced IR.
    let output_text = format!("{}", reduced.display_with_registry(&registry));
    let output_file = format!(
        "reduced_{}_{}.vixen-ir",
        format,
        ty.replace(['<', '>', ' '], "_")
    );
    std::fs::write(&output_file, &output_text).unwrap();
    eprintln!("[reduce-ir] wrote {output_file}");
    println!("{output_text}");
}

// ─── reduce-hir: minimize .vixen-hir reproducer ─────────────────────────────

fn cmd_reduce_hir(path: &str) {
    let source = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error: failed to read {path}: {e}");
        std::process::exit(1);
    });

    let module = kajit_hir_text::parse_hir(&source).unwrap_or_else(|e| {
        eprintln!("error: failed to parse HIR: {e}");
        std::process::exit(1);
    });

    // Verify the original triggers a panic
    if !hir_compile_panics_with_ssa(&module) {
        eprintln!("error: original HIR does not panic during compilation — nothing to reduce");
        std::process::exit(1);
    }
    eprintln!(
        "[reduce-hir] confirmed: original panics ({} statements)",
        count_stmts(&module)
    );

    let mut best = module;
    let fid = best.functions.iter().next().unwrap().0;
    let mut changed = true;
    while changed {
        changed = false;
        let func = &best.functions[fid];
        let paths = stmt_paths(&func.body);
        // Try removing each statement, last-first so indices stay valid
        for path_idx in (0..paths.len()).rev() {
            let mut candidate = best.clone();
            let func = &mut candidate.functions[fid];
            remove_stmt_at_path(&mut func.body, &paths[path_idx]);
            // Re-serialize and re-parse to validate
            let text = candidate.to_string();
            let Ok(reparsed) = kajit_hir_text::parse_hir(&text) else {
                continue;
            };
            if hir_compile_panics_with_ssa(&reparsed) {
                let old_count = count_stmts(&best);
                best = reparsed;
                let new_count = count_stmts(&best);
                eprintln!(
                    "[reduce-hir] removed stmt at {:?}: {} -> {} stmts",
                    paths[path_idx], old_count, new_count
                );
                changed = true;
                break; // restart from the beginning with the smaller module
            }
        }
    }

    let result = best.to_string();
    let output_path = path.replace(".vixen-hir", ".min.vixen-hir");
    std::fs::write(&output_path, &result).unwrap();
    eprintln!(
        "[reduce-hir] done: {} statements, written to {output_path}",
        count_stmts(&best)
    );
    println!("{result}");
}

fn hir_compile_panics_with_ssa(module: &kajit_hir::Module) -> bool {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let registry = kajit_ir::IntrinsicRegistry::empty();
        let pipeline_opts = kajit::PipelineOptions::builder()
            .compile_target(kajit::CompileTarget::Object)
            .build();
        let _ = kajit::compile_pipeline_from_hir_module(module, &registry, &pipeline_opts);
    }));
    match result {
        Ok(()) => false,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("");
            msg.contains("SSA validation failed")
        }
    }
}

fn count_stmts(module: &kajit_hir::Module) -> usize {
    module
        .functions
        .iter()
        .map(|(_, f)| count_stmts_in_block(&f.body))
        .sum()
}

fn count_stmts_in_block(block: &kajit_hir::Block) -> usize {
    let mut n = block.statements.len();
    for stmt in &block.statements {
        n += count_stmts_in_stmt(stmt);
    }
    n
}

fn count_stmts_in_stmt(stmt: &kajit_hir::Stmt) -> usize {
    use kajit_hir::StmtKind;
    match &stmt.kind {
        StmtKind::If {
            then_block,
            else_block,
            ..
        } => count_stmts_in_block(then_block) + else_block.as_ref().map_or(0, count_stmts_in_block),
        StmtKind::Loop { body, .. } => count_stmts_in_block(body),
        StmtKind::Match { arms, .. } => arms.iter().map(|a| count_stmts_in_block(&a.body)).sum(),
        _ => 0,
    }
}

/// Return paths (indices into nested blocks) for every statement.
fn stmt_paths(block: &kajit_hir::Block) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    collect_paths(block, &mut vec![], &mut result);
    result
}

fn collect_paths(block: &kajit_hir::Block, prefix: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
    for (i, stmt) in block.statements.iter().enumerate() {
        prefix.push(i);
        result.push(prefix.clone());
        // Recurse into sub-blocks
        use kajit_hir::StmtKind;
        match &stmt.kind {
            StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                collect_paths(then_block, prefix, result);
                if let Some(eb) = else_block {
                    collect_paths(eb, prefix, result);
                }
            }
            StmtKind::Loop { body, .. } => {
                collect_paths(body, prefix, result);
            }
            StmtKind::Match { arms, .. } => {
                for arm in arms {
                    collect_paths(&arm.body, prefix, result);
                }
            }
            _ => {}
        }
        prefix.pop();
    }
}

fn remove_stmt_at_path(block: &mut kajit_hir::Block, path: &[usize]) {
    if path.len() == 1 {
        if path[0] < block.statements.len() {
            block.statements.remove(path[0]);
        }
        return;
    }
    let idx = path[0];
    if idx >= block.statements.len() {
        return;
    }
    use kajit_hir::StmtKind;
    match &mut block.statements[idx].kind {
        StmtKind::If {
            then_block,
            else_block,
            ..
        } => {
            // Try then_block first; path[1] may refer to either sub-block
            if path[1] < then_block.statements.len() || path.len() > 2 {
                remove_stmt_at_path(then_block, &path[1..]);
            } else if let Some(eb) = else_block {
                remove_stmt_at_path(eb, &path[1..]);
            }
        }
        StmtKind::Loop { body, .. } => {
            remove_stmt_at_path(body, &path[1..]);
        }
        StmtKind::Match { arms, .. } => {
            for arm in arms {
                if path[1] < arm.body.statements.len() || path.len() > 2 {
                    remove_stmt_at_path(&mut arm.body, &path[1..]);
                    break;
                }
            }
        }
        _ => {}
    }
}

// ─── compile --reduce: minimize CFG-MIR reproducer ───────────────────────────

fn cmd_compile_reduce(format: &str, ty: &str, mode: &str) {
    use kajit_mir::opt::reduce;

    let kind = match format {
        "postcard" => kajit::DecoderKind::Postcard,
        other => {
            eprintln!("error: unknown format '{other}', expected 'postcard'");
            std::process::exit(1);
        }
    };

    let shape = resolve_shape(ty);
    let pipeline_opts = kajit::PipelineOptions::from_env();

    eprintln!("compiling {format} {ty} → pre-opt CFG-MIR...");
    let program = kajit::compile_pre_opt_cfg(shape, kind, &pipeline_opts);

    let initial_size = kajit_mir::ProgramSize::of(&program);
    eprintln!(
        "pre-opt CFG: {} blocks, {} insts, {} edges",
        initial_size.blocks, initial_size.insts, initial_size.edges
    );

    let input = make_test_input(format, ty);
    eprintln!("test input: {} bytes ({})", input.len(), encode_hex(&input));

    match mode {
        "differential" | "diff" => {
            eprintln!("error: differential reduction is not currently supported");
            eprintln!("use pass-based reduction instead: --reduce <pass_name>");
            std::process::exit(1);
        }
        other => {
            // Interpret as pass name(s) — SSA breakage mode
            let passes: Vec<&str> = other.split(',').map(|s| s.trim()).collect();
            for &p in &passes {
                if !reduce::ALL_PASS_NAMES.contains(&p) {
                    eprintln!("error: unknown pass or mode '{p}'");
                    eprintln!("modes: all/differential, or pass name(s)");
                    eprintln!("passes: {}", reduce::ALL_PASS_NAMES.join(", "));
                    std::process::exit(1);
                }
            }

            eprintln!("mode: SSA breakage after passes {:?}", passes);

            if !reduce::sequence_breaks_ssa(&program, &passes) {
                eprintln!("pass sequence does NOT break SSA — nothing to reduce");
                std::process::exit(0);
            }

            eprintln!("confirmed: SSA breaks after {:?}", passes);
            let predicate = reduce::predicate_sequence_breaks_ssa(
                &passes.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            );
            let result = reduce::reduce(&program, &*predicate);
            let reduced_text = format!("{}", result.program);

            let pass_label = passes.join("+");
            let output_path = format!("reduced_{format}_{ty}_{pass_label}.cfgmir");
            std::fs::write(&output_path, &reduced_text).unwrap_or_else(|e| {
                eprintln!("warning: failed to write {output_path}: {e}");
            });
            eprintln!(
                "reduction: {} → {} ({} steps, {} candidates)",
                initial_size.blocks,
                result
                    .program
                    .funcs
                    .iter()
                    .map(|f| f.blocks.iter().filter(|b| !b.dead).count())
                    .sum::<usize>(),
                result.steps_applied,
                result.candidates_tested
            );
            eprintln!("wrote: {output_path}");
            print!("{reduced_text}");
        }
    }
}

/// Generate a representative test input for a type in a given format.
fn make_test_input(format: &str, ty: &str) -> Vec<u8> {
    match format {
        "postcard" => match ty {
            // Postcard varints: encode 128 (needs 2 bytes: 0x80 0x01)
            "u32" | "u64" | "i32" | "i64" | "u16" | "i16" => vec![0x80, 0x01],
            "u8" | "i8" => vec![42],
            "bool" => vec![1],
            "String" | "string" => {
                // Length-prefixed: length=5, then "hello"
                vec![5, b'h', b'e', b'l', b'l', b'o']
            }
            _ => vec![0x80, 0x01], // default: 2-byte varint
        },
        _ => vec![0x80, 0x01],
    }
}

// ─── debug-diff: lockstep differential debugger ──────────────────────────────

fn cmd_debug_diff(format: &str, ty: &str, input_hex: &str) {
    let kind = match format {
        "postcard" => kajit::DecoderKind::Postcard,
        other => {
            eprintln!("error: unknown format '{other}'");
            std::process::exit(1);
        }
    };

    let shape = resolve_shape(ty);
    let pipeline_opts = kajit::PipelineOptions::from_env();

    // Phase 1: Compile and generate harness
    eprintln!("[debug-diff] compiling {format} {ty}...");
    let artifacts = kajit::compile_pipeline(shape, kind, &pipeline_opts);

    let output_size = shape.layout.sized_layout().map(|l| l.size()).unwrap_or(0);
    let output_dir = std::path::PathBuf::from("/tmp/kajit-harness");
    let base_name = format!("harness_{format}_{ty}");
    let listing_path = output_dir.join(format!("{base_name}.cfg-mir"));

    let dwarf = artifacts.decoder.build_standalone_dwarf(&listing_path);

    let known = kajit::intrinsics::known_intrinsics();
    let intrinsic_calls: Vec<_> = artifacts
        .intrinsic_call_sites
        .iter()
        .filter_map(|site| {
            let name = known
                .iter()
                .find(|(_, f)| f.0 == site.func.0)
                .map(|(name, _)| name.to_string())?;
            Some(kajit::harness::IntrinsicCallSite {
                code_offset: site.code_offset,
                baked_addr: site.func.0 as u64,
                symbol_name: name,
            })
        })
        .collect();

    let harness_input = kajit::harness::HarnessInput {
        code: artifacts.decoder.code(),
        entry_offset: artifacts.decoder.entry_offset(),
        output_size,
        dwarf,
        cfg_mir_lines: artifacts.decoder.cfg_mir_lines(),
        function_name: "kajit_decode",
        alloc_map: Some(&artifacts.alloc_map),
        intrinsic_calls,
        extern_addr_relocs: artifacts.extern_addr_relocs.clone(),
    };

    let exe_path = kajit::harness::generate_harness(&harness_input, &output_dir, &base_name)
        .unwrap_or_else(|e| {
            eprintln!("error generating harness: {e}");
            std::process::exit(1);
        });

    // Phase 3: Launch LLDB on the harness + run lockstep
    #[cfg(not(feature = "lldb"))]
    {
        let _ = (exe_path, artifacts, input_hex);
        eprintln!("error: debug-diff requires LLDB support");
        eprintln!("  rebuild with: cargo build -p kajit-cli");
        eprintln!("  or explicitly: cargo build -p kajit-cli --features lldb");
        std::process::exit(1);
    }

    #[cfg(feature = "lldb")]
    {
        eprintln!("[debug-diff] launching LLDB on {}...", exe_path.display());
        let mut debugger =
            lldb_debugger::LldbJitDebugger::launch(exe_path.to_str().unwrap(), input_hex)
                .unwrap_or_else(|e| {
                    eprintln!("error launching LLDB: {e}");
                    std::process::exit(1);
                });

        // Phase 4: Run lockstep
        let listing_lines = artifacts.decoder.cfg_mir_lines().to_vec();
        let total_ops: usize = artifacts
            .cfg_program
            .funcs
            .iter()
            .map(|f| f.blocks.iter().map(|b| b.insts.len() + 1).sum::<usize>())
            .sum();
        let dead_blocks: usize = artifacts
            .cfg_program
            .funcs
            .iter()
            .map(|f| f.blocks.iter().filter(|b| b.dead).count())
            .sum();
        eprintln!(
            "[debug-diff] listing: {} lines (decoder/cfg_program), {} total ops, {} dead blocks",
            listing_lines.len(),
            total_ops,
            dead_blocks
        );

        let input = parse_hex(input_hex);
        eprintln!(
            "[debug-diff] running lockstep (input: {} bytes)...\n",
            input.len()
        );

        let result = kajit::lockstep::run_lockstep(
            &artifacts.cfg_program,
            &input,
            &artifacts.location_map,
            &listing_lines,
            artifacts.backend_debug_info.as_ref(),
            artifacts.decoder.entry_offset(),
            &mut debugger,
            10_000,
        )
        .unwrap_or_else(|e| {
            eprintln!("error during lockstep: {e}");
            std::process::exit(1);
        });

        // Phase 5: Print result
        print!("{}", kajit::lockstep::format_result(&result));
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
