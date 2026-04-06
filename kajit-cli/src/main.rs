#[cfg(feature = "lldb")]
mod lldb_debugger;
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

    /// Compile a type and dump pipeline stages
    Compile {
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
        Command::Compile {
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
            uses_root_cursor_arg: artifacts.decoder.uses_root_cursor_arg(),
            alloc_map: Some(&artifacts.alloc_map),
            intrinsic_calls,
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
        let ir_text = &artifacts.ir_opt_timeline[0].1;
        let ir_registry = kajit::symbol_registry_for_shape(shape);
        match kajit_ir_text::parse_ir(ir_text, &ir_registry) {
            Ok(ir_func) => {
                let ir_result = kajit_ir::interpret::interpret(&ir_func, &input, output_size);
                match &ir_result.trap {
                    None => println!(
                        "  rvsdg out:   {} ({})",
                        encode_hex(&ir_result.output),
                        ir_result.output.len()
                    ),
                    Some(t) => println!("  rvsdg:       TRAP ({:?})", t.code),
                }
            }
            Err(e) => println!("  rvsdg:       parse error: {e}"),
        }

        // Run CFG-MIR interpreter (post-linearization)
        let interp_result = kajit_mir::opt::reduce::interpret(&artifacts.cfg_program, &input);
        match &interp_result {
            kajit_mir::opt::reduce::InterpOutcome::Returned(bytes) => {
                println!("  interp out:  {} ({})", encode_hex(bytes), bytes.len())
            }
            kajit_mir::opt::reduce::InterpOutcome::Trapped(trap) => {
                println!(
                    "  interp:      TRAP ({:?} at offset {})",
                    trap.code, trap.offset
                )
            }
            kajit_mir::opt::reduce::InterpOutcome::TimedOut => {
                println!("  interp:      TIMEOUT")
            }
        }

        // Run JIT (may hang on buggy programs — run after interpreter)
        let jit_result = kajit::deserialize_raw(&artifacts.decoder, &input, output_size);
        match &jit_result {
            Ok(bytes) => println!("  jit output:  {} ({})", encode_hex(bytes), bytes.len()),
            Err(e) => println!("  jit error:   {e}"),
        }

        // Compare
        use kajit_mir::opt::reduce::InterpOutcome;
        match (&jit_result, &interp_result) {
            (Ok(jit), InterpOutcome::Returned(interp)) if jit == interp => {
                println!("  match:       YES");
            }
            (Ok(jit), InterpOutcome::Returned(interp)) => {
                println!("  match:       NO — DIVERGENCE");
                // Show byte-by-byte diff
                let max_len = jit.len().max(interp.len());
                for i in 0..max_len {
                    let j = jit.get(i).copied();
                    let r = interp.get(i).copied();
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
        "Vec<u8>" => Vec::<u8>::SHAPE,
        "Vec<u32>" => Vec::<u32>::SHAPE,
        "Vec<u64>" => Vec::<u64>::SHAPE,
        "Vec<String>" => Vec::<String>::SHAPE,
        "Option<u32>" => Option::<u32>::SHAPE,
        "Option<String>" => Option::<String>::SHAPE,
        "(u32, u32)" => <(u32, u32)>::SHAPE,
        other => {
            eprintln!("error: unknown type '{other}'");
            eprintln!("supported: u8 u16 u32 u64 i8 i16 i32 i64 bool String");
            eprintln!("          Vec<u8> Vec<u32> Vec<u64> Vec<String>");
            eprintln!("          Option<u32> Option<String> (u32, u32)");
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
        uses_root_cursor_arg: artifacts.decoder.uses_root_cursor_arg(),
        alloc_map: Some(&artifacts.alloc_map),
        intrinsic_calls,
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
