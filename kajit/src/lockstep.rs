#![allow(dead_code)]
//! Lockstep differential debugger.
//!
//! Drives the CFG-MIR interpreter and a JIT process (via LLDB) simultaneously,
//! comparing vreg values after each CFG-MIR operation and stopping on the first
//! divergence.

use crate::harness::{LocationMap, LocationTracker, VRegLocation};
use kajit_lir::{BinOpKind, LinearOp};

/// A divergence found by the lockstep debugger.
#[derive(Debug, Clone)]
pub struct Divergence {
    /// Which interpreter step triggered the divergence.
    pub step: usize,
    /// The DWARF line number (= CFG-MIR listing line).
    pub dwarf_line: u32,
    /// The CFG-MIR source line text.
    pub source_line: String,
    /// Per-vreg comparison results at this point.
    pub vreg_diffs: Vec<VRegDiff>,
}

/// Comparison of a single vreg between interpreter and JIT.
#[derive(Debug, Clone)]
pub struct VRegDiff {
    pub vreg_index: u32,
    pub interpreter_value: u64,
    pub jit_value: Option<u64>,
    pub jit_location: VRegLocation,
    pub matches: bool,
}

/// Trait for reading JIT process state. Implemented by LLDB backend.
pub trait JitDebugger {
    /// Step until the DWARF source line changes. Returns the new line number.
    fn step_to_next_source_line(&mut self) -> Result<u32, DebugError>;

    /// Step a single machine instruction, stepping over calls.
    fn step_instruction_over(&mut self) -> Result<(), DebugError>;

    /// Read a general-purpose register by index (0=x0, 1=x1, ..., 28=x28).
    fn read_register(&self, preg: u8) -> Result<u64, DebugError>;

    /// Read memory at the given address.
    fn read_memory(&self, addr: u64, size: usize) -> Result<Vec<u8>, DebugError>;

    /// Read the stack pointer.
    fn read_sp(&self) -> Result<u64, DebugError>;

    /// Get the current program counter.
    fn read_pc(&self) -> Result<u64, DebugError>;

    /// Check if the process has exited.
    fn has_exited(&self) -> bool;

    /// Get the current DWARF source line number.
    fn current_source_line(&self) -> Result<u32, DebugError>;

    /// Get disassembly around the current PC (a few instructions before/after).
    fn disassemble_around_pc(&self, context: usize) -> Result<String, DebugError>;
}

#[derive(Debug)]
pub enum DebugError {
    /// Process exited normally with exit code.
    ProcessExited(i32),
    /// Process killed by signal (SIGSEGV=11, SIGBUS=10, SIGABRT=6, etc.)
    ProcessSignaled(i32),
    LldbError(String),
    Io(std::io::Error),
}

impl std::fmt::Display for DebugError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DebugError::ProcessExited(code) => write!(f, "process exited with code {code}"),
            DebugError::ProcessSignaled(sig) => {
                let name = match sig {
                    6 => "SIGABRT",
                    10 => "SIGBUS",
                    11 => "SIGSEGV",
                    _ => "unknown",
                };
                write!(f, "process killed by signal {sig} ({name})")
            }
            DebugError::LldbError(msg) => write!(f, "lldb: {msg}"),
            DebugError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

/// Result of a lockstep debugging session.
#[derive(Debug)]
pub struct LockstepResult {
    /// Total steps executed before stopping.
    pub steps: usize,
    /// The divergence found, if any.
    pub divergence: Option<Divergence>,
    /// Whether the session completed without divergence.
    pub completed: bool,
}

/// Run the lockstep debugger.
///
/// Steps both the interpreter and JIT debugger, comparing vreg values at each
/// CFG-MIR operation. Stops at the first divergence or when both complete.
pub fn run_lockstep(
    program: &kajit_mir::cfg_mir::Program,
    input: &[u8],
    location_map: &LocationMap,
    listing_lines: &[String],
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    entry_offset: usize,
    debugger: &mut dyn JitDebugger,
    max_steps: usize,
) -> Result<LockstepResult, DebugError> {
    // Read the JIT's runtime bases at the breakpoint. Root container decoders
    // use the extra cursor-ref argument in x2; plain decoders still materialize
    // cursor/end in x19/x20 after the prologue.
    let jit_output_ptr = debugger.read_register(21)?; // x21
    let jit_root_cursor_arg = if !program.funcs[0].data_args.is_empty() {
        Some(debugger.read_register(2)?)
    } else {
        None
    };
    let (jit_cursor, jit_input_end) = if let Some(root_cursor_arg) = jit_root_cursor_arg {
        let input_base = read_u64(debugger, root_cursor_arg)?;
        let input_len = read_u64(debugger, root_cursor_arg + 8)?;
        (input_base, input_base.wrapping_add(input_len))
    } else {
        (debugger.read_register(19)?, debugger.read_register(20)?)
    };

    eprintln!(
        "[lockstep] JIT base addrs: cursor=0x{:x}, end=0x{:x}, out=0x{:x}",
        jit_cursor, jit_input_end, jit_output_ptr
    );

    // Set up interpreter with the JIT's base addresses so pointer values match
    let mut session = kajit_mir::DebuggerSession::new(program, input)
        .map_err(|e| DebugError::LldbError(format!("interpreter init: {e}")))?;
    session.input_base_addr = Some(jit_cursor);
    session.input_end_addr = Some(jit_input_end);
    session.output_base_addr = Some(jit_output_ptr);
    if let Some(root_cursor_arg) = jit_root_cursor_arg {
        session.set_root_cursor_arg_addr(root_cursor_arg);
    }

    // Build bidirectional mapping: DWARF line ↔ (block, inst_index, is_term)
    let func = &program.funcs[0];
    // Map (block_id, is_terminator, inst_index_if_not_term) → DWARF line
    // For terminators, inst_index doesn't matter — we key by (block, true, 0)
    let mut op_to_line: std::collections::HashMap<(u32, bool, usize), u32> =
        std::collections::HashMap::new();
    let mut next_line = 1u32;
    for block in &func.blocks {
        // Include ALL blocks (even dead ones) to match build_debug_line_maps
        for (inst_idx, _) in block.insts.iter().enumerate() {
            op_to_line.insert((block.id.0, false, inst_idx), next_line);
            next_line += 1;
        }
        // Terminator: key with is_term=true, inst_idx=0 (ignored)
        op_to_line.insert((block.id.0, true, 0), next_line);
        next_line += 1;
    }

    let mut jit_steps = 0;
    let entry_pc = debugger.read_pc()?;
    let code_line_ranges = build_code_line_ranges(backend_debug_info, entry_offset);
    let mut prev_dwarf_line =
        current_line_from_pc(debugger, entry_pc, &code_line_ranges).unwrap_or(1);

    // History: vreg_index → (last_verified_value, at_line, at_step)
    let mut verified: std::collections::HashMap<u32, (u64, u32, usize)> =
        std::collections::HashMap::new();
    let mut location_tracker = LocationTracker::new(location_map, program);

    loop {
        if jit_steps >= max_steps {
            eprintln!("[lockstep] max steps ({max_steps}) reached");
            return Ok(LockstepResult {
                steps: jit_steps,
                divergence: None,
                completed: false,
            });
        }

        if debugger.has_exited() {
            return Ok(LockstepResult {
                steps: jit_steps,
                divergence: None,
                completed: true,
            });
        }

        // Step JIT one source line
        let dwarf_line = match step_to_next_mapped_line(
            debugger,
            entry_pc,
            &code_line_ranges,
            prev_dwarf_line,
        ) {
            Ok(line) => line,
            Err(DebugError::ProcessExited(code)) => {
                return handle_jit_exit(
                    &mut session,
                    &op_to_line,
                    listing_lines,
                    jit_steps,
                    prev_dwarf_line,
                    &format!("exited with code {code}"),
                );
            }
            Err(DebugError::ProcessSignaled(sig)) => {
                let sig_name = match sig {
                    6 => "SIGABRT",
                    10 => "SIGBUS",
                    11 => "SIGSEGV",
                    _ => "signal",
                };
                return handle_jit_exit(
                    &mut session,
                    &op_to_line,
                    listing_lines,
                    jit_steps,
                    prev_dwarf_line,
                    &format!("killed by {} (signal {})", sig_name, sig),
                );
            }
            Err(e) => return Err(e),
        };
        jit_steps += 1;

        // The JIT was at prev_dwarf_line, stepped to dwarf_line.
        // That means it executed the op at prev_dwarf_line.
        let executed_line = prev_dwarf_line;
        prev_dwarf_line = dwarf_line;

        // Step interpreter until it reaches dwarf_line (where the JIT is now).
        // We sync by (block, inst) position, not just line number, because
        // DWARF lines follow emission order while the interpreter follows
        // execution order — these differ in unrolled code.
        //
        // Strategy: step the interpreter, capturing the event when its line
        // matches executed_line. Stop when its line matches dwarf_line.
        let mut last_event = None;
        let mut synced = false;
        for _ in 0..500 {
            let pre_loc = session.state().location;
            let pre_line = loc_to_line(&op_to_line, &pre_loc);

            if pre_line == executed_line && last_event.is_none() {
                let event = session
                    .step_forward()
                    .map_err(|e| DebugError::LldbError(format!("interpreter step: {e}")))?;
                last_event = Some(event.clone());
                if event.returned {
                    break;
                }
                continue;
            }

            if pre_line == dwarf_line {
                synced = true;
                break;
            }

            // If we already captured last_event and overshot past dwarf_line,
            // the interpreter and JIT may be visiting blocks in different order.
            // In that case, just step until we hit dwarf_line or the interpreter
            // catches up.
            let event = session
                .step_forward()
                .map_err(|e| DebugError::LldbError(format!("interpreter step: {e}")))?;

            // Also capture if this step happens to hit executed_line
            let ev_line = loc_to_line(&op_to_line, &event.location_before);
            if ev_line == executed_line && last_event.is_none() {
                last_event = Some(event.clone());
            }

            if event.returned {
                break;
            }
        }

        if !synced && !session.state().returned {
            let cur_loc = session.state().location;
            let cur_line = loc_to_line(&op_to_line, &cur_loc);
            let disasm = debugger.disassemble_around_pc(4).unwrap_or_default();
            let line_text = |n: u32| -> String {
                if n > 0 && (n as usize) <= listing_lines.len() {
                    listing_lines[n as usize - 1].clone()
                } else {
                    format!("<line {n}>")
                }
            };

            let source_line = format!(
                "\
SYNC FAILURE at JIT step {jit_steps}

  JIT executed line {executed_line}: {}
  JIT now at line {dwarf_line}: {}
  interpreter stuck at line {cur_line} (b{} inst[{}] term={}): {}

  machine code at JIT position:
{disasm}",
                line_text(executed_line),
                line_text(dwarf_line),
                cur_loc.block.index(),
                cur_loc.next_inst_index,
                cur_loc.at_terminator,
                line_text(cur_line),
            );

            return Ok(LockstepResult {
                steps: jit_steps,
                divergence: Some(Divergence {
                    step: jit_steps,
                    dwarf_line: executed_line,
                    source_line,
                    vreg_diffs: Vec::new(),
                }),
                completed: false,
            });
        }

        let Some(event) = last_event else {
            let cur_loc = session.state().location;
            let cur_line = loc_to_line(&op_to_line, &cur_loc);
            let disasm = debugger.disassemble_around_pc(4).unwrap_or_default();
            let line_text = |n: u32| -> String {
                if n > 0 && (n as usize) <= listing_lines.len() {
                    listing_lines[n as usize - 1].clone()
                } else {
                    format!("<line {n}>")
                }
            };

            let source_line = format!(
                "\
LOCKSTEP DESYNC at JIT step {jit_steps}

  JIT executed line {executed_line}: {}
  JIT now at line {dwarf_line}: {}
  interpreter at line {cur_line} (b{} inst[{}] term={}): {}

  The interpreter could not find line {executed_line} — the JIT and interpreter
  are visiting different blocks. This is either a control flow bug or a
  DWARF line numbering mismatch.

  machine code at JIT position:
{disasm}",
                line_text(executed_line),
                line_text(dwarf_line),
                cur_loc.block.index(),
                cur_loc.next_inst_index,
                cur_loc.at_terminator,
                line_text(cur_line),
            );

            return Ok(LockstepResult {
                steps: jit_steps,
                divergence: Some(Divergence {
                    step: jit_steps,
                    dwarf_line: executed_line,
                    source_line,
                    vreg_diffs: Vec::new(),
                }),
                completed: false,
            });
        };
        let state = session.state();
        let func = &program.funcs[0];
        let loc = &event.location_before;
        let (def_vreg, use_vregs, _) = op_def_uses_and_kind(func, loc);

        // After stepping, check where BOTH sides ended up NEXT.
        // The JIT is now at `dwarf_line`. The interpreter is at its next op.
        // If these don't match, we have a CONTROL FLOW DIVERGENCE.
        let interp_next_loc = session.state().location;
        let interp_next_line = op_to_line
            .get(&(
                interp_next_loc.block.0,
                interp_next_loc.at_terminator,
                if interp_next_loc.at_terminator {
                    0
                } else {
                    interp_next_loc.next_inst_index
                },
            ))
            .copied()
            .unwrap_or(0);

        location_tracker.observe_step(location_map, func, executed_line, loc, &interp_next_loc);

        if interp_next_line != dwarf_line && !state.returned && dwarf_line != 0 {
            // CONTROL FLOW DIVERGENCE — build rich diagnostic
            let jit_pc = debugger.read_pc()?;
            let disasm = debugger.disassemble_around_pc(4).unwrap_or_default();

            let line = |n: u32| -> String {
                if n > 0 && (n as usize) <= listing_lines.len() {
                    listing_lines[n as usize - 1].clone()
                } else {
                    format!("<line {n}>")
                }
            };

            // Look up the terminator and its edge targets from the CFG
            let term_info = {
                let loc = &event.location_before;
                let block = &func.blocks[loc.block.index()];
                let term = &func.terms[block.term.index()];
                let mut info = format!("  terminator: {:?}\n", term);
                for &edge_id in &block.succs {
                    let edge = &func.edges[edge_id.index()];
                    let _target = &func.blocks[edge.to.index()];
                    let target_line = op_to_line
                        .get(&(edge.to.0, false, 0))
                        .or_else(|| op_to_line.get(&(edge.to.0, true, 0)))
                        .copied()
                        .unwrap_or(0);
                    info.push_str(&format!(
                        "  edge e{}: b{} → b{} (line {}) args={}\n",
                        edge_id.index(),
                        edge.from.index(),
                        edge.to.index(),
                        target_line,
                        edge.args.len()
                    ));
                }
                info
            };

            let mut vreg_diffs = Vec::new();
            if let (Some(dst), Some(location)) = (
                def_vreg,
                def_vreg.and_then(|d| location_tracker.location_for(location_map, d.index() as u32)),
            ) {
                let sp = debugger.read_sp()?;
                let iv = if dst.index() < state.vregs.len() {
                    state.vregs[dst.index()]
                } else {
                    0
                };
                let jv = read_vreg_from_jit(debugger, location, sp)?;
                vreg_diffs.push(VRegDiff {
                    vreg_index: dst.index() as u32,
                    interpreter_value: iv,
                    jit_value: jv,
                    jit_location: location.clone(),
                    matches: jv == Some(iv),
                });
            }

            let source_line = format!(
                "\
CONTROL FLOW DIVERGENCE at step {jit_steps}

  last op (line {executed_line}): {}
{term_info}
  JIT went to:         line {dwarf_line} (pc=0x{jit_pc:x}): {}
  interpreter went to: line {interp_next_line}: {}

  machine code:
{disasm}",
                line(executed_line),
                line(dwarf_line),
                line(interp_next_line),
            );

            return Ok(LockstepResult {
                steps: jit_steps,
                divergence: Some(Divergence {
                    step: jit_steps,
                    dwarf_line: executed_line,
                    source_line,
                    vreg_diffs,
                }),
                completed: false,
            });
        }

        if should_skip_fused_compare_value_check(func, loc, def_vreg)
            || should_skip_branch_predicate_value_check(func, loc)
            || should_skip_process_local_intrinsic_return_check(func, loc)
        {
            continue;
        }

        // Compare: either the defined vreg, or for store ops, the source vreg
        let compare_vregs: Vec<kajit_ir::VReg> = if let Some(dst) = def_vreg {
            vec![dst]
        } else if !use_vregs.is_empty() {
            // No def — this is a store/side-effect op. Compare its use vregs.
            use_vregs.clone()
        } else {
            Vec::new()
        };

        if let Some(&dst) = compare_vregs.first() {
            let dst_idx = dst.index() as u32;
            // Use per-line location lookup: at call sites, caller-saved registers
            // are clobbered. location_at returns None for clobbered locations.
            let location = location_tracker.location_for(location_map, dst_idx);
            if let Some(location) = location {
                let interp_value = if dst.index() < state.vregs.len() {
                    state.vregs[dst.index()]
                } else {
                    continue;
                };

                let sp = debugger.read_sp()?;
                let jit_value = read_vreg_from_jit(debugger, location, sp)?;

                if jit_value != Some(interp_value) {
                    let mut vreg_diffs = vec![VRegDiff {
                        vreg_index: dst_idx,
                        interpreter_value: interp_value,
                        jit_value,
                        jit_location: location.clone(),
                        matches: false,
                    }];
                    for use_vreg in &use_vregs {
                        let use_idx = use_vreg.index() as u32;
                        if let Some(use_loc) =
                            location_tracker.location_for(location_map, use_idx)
                        {
                            let use_interp = if use_vreg.index() < state.vregs.len() {
                                state.vregs[use_vreg.index()]
                            } else {
                                0
                            };
                            let use_jit = read_vreg_from_jit(debugger, use_loc, sp)?;
                            vreg_diffs.push(VRegDiff {
                                vreg_index: use_idx,
                                interpreter_value: use_interp,
                                jit_value: use_jit,
                                jit_location: use_loc.clone(),
                                matches: use_jit == Some(use_interp),
                            });
                        }
                    }
                    let disasm = debugger.disassemble_around_pc(4).unwrap_or_default();
                    let line_text = |n: u32| -> String {
                        if n > 0 && (n as usize) <= listing_lines.len() {
                            listing_lines[n as usize - 1].clone()
                        } else {
                            format!("<line {n}>")
                        }
                    };

                    // Build rich diagnostic: history, register sharing, clobber trail
                    let mut diag = String::new();
                    diag.push_str(&format!(
                        "VALUE DIVERGENCE at step {jit_steps}, line {executed_line}\n\n"
                    ));
                    diag.push_str(&format!("  op: {}\n", line_text(executed_line)));

                    // Show the diverging vreg
                    let reg_name = match location {
                        VRegLocation::Register(p) => LocationMap::reg_name(*p),
                        _ => "stk",
                    };
                    diag.push_str(&format!(
                        "\n  v{dst_idx} ({reg_name}): interpreter={interp_value}, jit={}\n",
                        jit_value.map(|v| v.to_string()).unwrap_or("???".into())
                    ));

                    // Show last verified value for this vreg
                    if let Some(&(last_val, last_line, last_step)) = verified.get(&dst_idx) {
                        diag.push_str(&format!(
                            "  v{dst_idx} last verified: {} at line {} (step {})\n",
                            last_val, last_line, last_step
                        ));
                    } else {
                        diag.push_str(&format!("  v{dst_idx} was NEVER verified before\n"));
                    }

                    // Show what else shares this physical register
                    if let VRegLocation::Register(preg) = location {
                        let sharing: Vec<_> = location_map
                            .static_locations
                            .iter()
                            .filter(|(k, v)| {
                                **k != dst_idx
                                    && matches!(v, VRegLocation::Register(p) if p == preg)
                            })
                            .map(|(k, _)| *k)
                            .collect();
                        if !sharing.is_empty() {
                            diag.push_str(&format!(
                                "\n  other vregs sharing {}: ",
                                LocationMap::reg_name(*preg)
                            ));
                            for (i, v) in sharing.iter().enumerate() {
                                if i > 0 {
                                    diag.push_str(", ");
                                }
                                if let Some(&(val, line, _step)) = verified.get(v) {
                                    diag.push_str(&format!("v{}(={} @line{})", v, val, line));
                                } else {
                                    diag.push_str(&format!("v{}", v));
                                }
                            }
                            diag.push('\n');

                            // Show which sharing vreg has the JIT's current value
                            if let Some(jv) = jit_value {
                                for &v in &sharing {
                                    if let Some(&(val, line, _)) = verified.get(&v) {
                                        if val == jv {
                                            diag.push_str(&format!(
                                                "\n  SUSPECT: v{} had value {} at line {} — same as JIT's current {}\n",
                                                v, val, line, reg_name
                                            ));
                                            diag.push_str(&format!(
                                                "  → {} was NOT updated between line {} and line {}\n",
                                                reg_name, line, executed_line
                                            ));
                                            diag.push_str("  → likely a missing phi/copy move on the edge into this block\n");
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Show use vregs
                    if !vreg_diffs.is_empty() {
                        diag.push_str("\n  inputs:\n");
                        for d in &vreg_diffs[1..] {
                            let m = if d.matches { "OK" } else { "MISMATCH" };
                            let loc_str = match &d.jit_location {
                                VRegLocation::Register(p) => LocationMap::reg_name(*p).to_string(),
                                VRegLocation::StackSlot(o) => format!("[sp+{o}]"),
                                VRegLocation::Constant(v) => format!("const({v})"),
                            };
                            diag.push_str(&format!(
                                "    v{}({}): interp={}, jit={} {}\n",
                                d.vreg_index,
                                loc_str,
                                d.interpreter_value,
                                d.jit_value.map(|v| v.to_string()).unwrap_or("???".into()),
                                m
                            ));
                        }
                    }

                    diag.push_str(&format!("\n  machine code:\n{disasm}\n"));

                    let source_line = diag;
                    return Ok(LockstepResult {
                        steps: jit_steps,
                        divergence: Some(Divergence {
                            step: jit_steps,
                            dwarf_line: executed_line,
                            source_line,
                            vreg_diffs,
                        }),
                        completed: false,
                    });
                }

                verified.insert(dst_idx, (interp_value, executed_line, jit_steps));

                let reg_name = match location {
                    VRegLocation::Register(p) => LocationMap::reg_name(*p),
                    _ => "stk",
                };
                eprintln!(
                    "[lockstep] line {executed_line}: v{}({}) = {} OK",
                    dst_idx, reg_name, interp_value
                );
            } else if location_map.call_lines.contains(&executed_line) {
                // Vreg is in a clobbered register at a call site — skip comparison
                eprintln!(
                    "[lockstep] line {executed_line}: v{dst_idx} skipped (call-clobbered register)"
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CodeLineRange {
    start: u64,
    end: u64,
    line: u32,
}

fn build_code_line_ranges(
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
    entry_offset: usize,
) -> Vec<CodeLineRange> {
    let Some(backend_debug_info) = backend_debug_info else {
        return Vec::new();
    };

    let mut ranges: Vec<_> = backend_debug_info
        .op_infos
        .iter()
        .flat_map(|op| {
            op.code_ranges.iter().filter_map(|range| {
                let start = range.start_offset as usize;
                let end = range.end_offset as usize;
                if end <= start || end <= entry_offset {
                    return None;
                }
                Some(CodeLineRange {
                    start: start.saturating_sub(entry_offset) as u64,
                    end: end.saturating_sub(entry_offset) as u64,
                    line: op.line,
                })
            })
        })
        .collect();
    ranges.sort_by_key(|range| (range.start, range.end, range.line));
    ranges
}

fn current_line_from_pc(
    debugger: &dyn JitDebugger,
    entry_pc: u64,
    code_line_ranges: &[CodeLineRange],
) -> Result<u32, DebugError> {
    let pc = debugger.read_pc()?;
    let offset = pc.saturating_sub(entry_pc);
    if let Some(range) = code_line_ranges
        .iter()
        .find(|range| offset >= range.start && offset < range.end)
    {
        return Ok(range.line);
    }
    Ok(debugger.current_source_line().unwrap_or(0))
}

fn step_to_next_mapped_line(
    debugger: &mut dyn JitDebugger,
    entry_pc: u64,
    code_line_ranges: &[CodeLineRange],
    start_line: u32,
) -> Result<u32, DebugError> {
    for _ in 0..4096 {
        debugger.step_instruction_over()?;
        let line = current_line_from_pc(debugger, entry_pc, code_line_ranges)?;
        if line != 0 && line != start_line {
            return Ok(line);
        }
    }

    Err(DebugError::LldbError(format!(
        "instruction stepping did not leave mapped line {start_line}"
    )))
}

fn read_u64(debugger: &dyn JitDebugger, addr: u64) -> Result<u64, DebugError> {
    let bytes = debugger.read_memory(addr, 8)?;
    if bytes.len() != 8 {
        return Err(DebugError::LldbError(format!(
            "short read at 0x{addr:x}: expected 8 bytes, got {}",
            bytes.len()
        )));
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes);
    Ok(u64::from_le_bytes(raw))
}

fn should_skip_fused_compare_value_check(
    func: &kajit_mir::cfg_mir::Function,
    loc: &kajit_mir::ProgramLocation,
    def_vreg: Option<kajit_ir::VReg>,
) -> bool {
    let Some(def_vreg) = def_vreg else {
        return false;
    };
    if loc.at_terminator {
        return false;
    }
    let Some(block) = func.blocks.get(loc.block.index()) else {
        return false;
    };
    if loc.next_inst_index + 1 != block.insts.len() {
        return false;
    }
    let Some(inst_id) = block.insts.get(loc.next_inst_index) else {
        return false;
    };
    let Some(inst) = func.insts.get(inst_id.index()) else {
        return false;
    };
    let is_compare = matches!(
        inst.op,
        LinearOp::BinOp {
            op: BinOpKind::CmpEq
                | BinOpKind::CmpNe
                | BinOpKind::CmpLt
                | BinOpKind::CmpLe
                | BinOpKind::CmpGt
                | BinOpKind::CmpGe,
            ..
        }
    );
    if !is_compare {
        return false;
    }
    matches!(
        &func.terms[block.term.index()],
        kajit_mir::cfg_mir::Terminator::BranchIf { cond, .. }
            | kajit_mir::cfg_mir::Terminator::BranchIfZero { cond, .. }
            if *cond == def_vreg
    )
}

fn should_skip_branch_predicate_value_check(
    func: &kajit_mir::cfg_mir::Function,
    loc: &kajit_mir::ProgramLocation,
) -> bool {
    if !loc.at_terminator {
        return false;
    }
    let Some(block) = func.blocks.get(loc.block.index()) else {
        return false;
    };
    matches!(
        &func.terms[block.term.index()],
        kajit_mir::cfg_mir::Terminator::BranchIf { .. }
            | kajit_mir::cfg_mir::Terminator::BranchIfZero { .. }
    )
}

fn should_skip_process_local_intrinsic_return_check(
    func: &kajit_mir::cfg_mir::Function,
    loc: &kajit_mir::ProgramLocation,
) -> bool {
    if loc.at_terminator {
        return false;
    }
    let Some(block) = func.blocks.get(loc.block.index()) else {
        return false;
    };
    let Some(inst_id) = block.insts.get(loc.next_inst_index) else {
        return false;
    };
    let Some(inst) = func.insts.get(inst_id.index()) else {
        return false;
    };
    let intrinsic = match inst.op {
        LinearOp::CallIntrinsic { func, dst, .. } if dst.is_some() => Some(func),
        LinearOp::CallPure { func, .. } | LinearOp::CallEffect { func, .. } => Some(func),
        _ => None,
    };
    intrinsic.is_some_and(is_process_local_pointer_intrinsic)
}

fn is_process_local_pointer_intrinsic(func: kajit_ir::IntrinsicFn) -> bool {
    let f = func.0;
    let alloc_persistent = crate::intrinsics::kajit_alloc_persistent as *const () as usize;
    let alloc_transient = crate::intrinsics::kajit_alloc_transient as *const () as usize;
    let vec_alloc = crate::intrinsics::kajit_vec_alloc as *const () as usize;
    let vec_grow = crate::intrinsics::kajit_vec_grow as *const () as usize;
    let map_build = crate::intrinsics::kajit_map_build as *const () as usize;
    let string_alloc =
        crate::intrinsics::kajit_postcard_validate_and_alloc_string as *const () as usize;
    let string_copy = crate::intrinsics::kajit_string_validate_alloc_copy as *const () as usize;
    matches!(
        f,
        x if x == alloc_persistent
            || x == alloc_transient
            || x == vec_alloc
            || x == vec_grow
            || x == map_build
            || x == string_alloc
            || x == string_copy
    )
}

/// Look up the DWARF line for an interpreter ProgramLocation.
fn loc_to_line(
    map: &std::collections::HashMap<(u32, bool, usize), u32>,
    loc: &kajit_mir::ProgramLocation,
) -> u32 {
    map.get(&(
        loc.block.0,
        loc.at_terminator,
        if loc.at_terminator {
            0
        } else {
            loc.next_inst_index
        },
    ))
    .copied()
    .unwrap_or(0)
}

/// Handle JIT process exit: run interpreter to see how far behind it is.
/// If the interpreter needs more steps, that's a divergence.
fn handle_jit_exit(
    session: &mut kajit_mir::DebuggerSession,
    op_to_line: &std::collections::HashMap<(u32, bool, usize), u32>,
    listing_lines: &[String],
    jit_steps: usize,
    last_dwarf_line: u32,
    exit_reason: &str,
) -> Result<LockstepResult, DebugError> {
    let mut interp_steps = 0;
    loop {
        let ev = session
            .step_forward()
            .map_err(|e| DebugError::LldbError(format!("interpreter step: {e}")))?;
        interp_steps += 1;
        if ev.returned || interp_steps > 10000 {
            break;
        }
    }
    let final_state = session.state();
    let interp_loc = final_state.location;
    let interp_line = loc_to_line(op_to_line, &interp_loc);

    let line_text = |n: u32| -> String {
        if n > 0 && (n as usize) <= listing_lines.len() {
            listing_lines[n as usize - 1].clone()
        } else {
            format!("<line {n}>")
        }
    };

    // If interpreter also returned quickly (≤2 extra steps), it's a genuine match
    if final_state.returned && interp_steps <= 2 {
        eprintln!(
            "[lockstep] both completed: JIT {} after {} steps, interpreter needed {} extra steps",
            exit_reason, jit_steps, interp_steps
        );
        return Ok(LockstepResult {
            steps: jit_steps,
            divergence: None,
            completed: true,
        });
    }

    // JIT exited but interpreter still had work — divergence
    let source_line = format!(
        "\
JIT EARLY EXIT at step {jit_steps}

  JIT {exit_reason} at line {last_dwarf_line}: {last_op}
  interpreter needed {interp_steps} more steps (returned={returned}, at line {interp_line}: {interp_op})

  The JIT exited while the interpreter still had {interp_steps} operations to execute.
  This means the JIT skipped a section of the program (likely an entire loop body).",
        last_op = line_text(last_dwarf_line),
        returned = final_state.returned,
        interp_op = line_text(interp_line),
    );

    Ok(LockstepResult {
        steps: jit_steps,
        divergence: Some(Divergence {
            step: jit_steps,
            dwarf_line: last_dwarf_line,
            source_line,
            vreg_diffs: Vec::new(),
        }),
        completed: false,
    })
}

/// Read a vreg's value from the JIT process.
fn read_vreg_from_jit(
    debugger: &dyn JitDebugger,
    location: &VRegLocation,
    sp: u64,
) -> Result<Option<u64>, DebugError> {
    match location {
        VRegLocation::Register(preg) => Ok(debugger.read_register(*preg).ok()),
        VRegLocation::StackSlot(offset) => {
            let addr = sp + *offset as u64;
            Ok(debugger
                .read_memory(addr, 8)
                .ok()
                .map(|b| u64::from_le_bytes(b[..8].try_into().unwrap())))
        }
        VRegLocation::Constant(val) => Ok(Some(*val)),
    }
}

/// Extract def vreg, use vregs, and whether this op produces a pointer value.
fn op_def_uses_and_kind(
    func: &kajit_mir::cfg_mir::Function,
    loc: &kajit_mir::ProgramLocation,
) -> (Option<kajit_ir::VReg>, Vec<kajit_ir::VReg>, bool) {
    use kajit_lir::LinearOp;
    use kajit_mir::cfg_mir::OperandKind;

    let block = &func.blocks[loc.block.index()];

    if loc.at_terminator {
        let term = &func.terms[block.term.index()];
        let uses = match term {
            kajit_mir::cfg_mir::Terminator::BranchIf { cond, .. }
            | kajit_mir::cfg_mir::Terminator::BranchIfZero { cond, .. } => vec![*cond],
            kajit_mir::cfg_mir::Terminator::JumpTable { predicate, .. } => vec![*predicate],
            _ => Vec::new(),
        };
        return (None, uses, false);
    }

    if loc.next_inst_index >= block.insts.len() {
        return (None, Vec::new(), false);
    }

    let inst_id = block.insts[loc.next_inst_index];
    let inst = &func.insts[inst_id.index()];

    let mut def = None;
    let mut uses = Vec::new();
    for op in &inst.operands {
        match op.kind {
            OperandKind::Def => def = Some(op.vreg),
            OperandKind::Use => uses.push(op.vreg),
        }
    }

    // Ops that produce pointer values (interpreter uses different base addresses)
    let is_pointer = matches!(
        inst.op,
        LinearOp::SaveCursor { .. }
            | LinearOp::SaveInputEnd { .. }
            | LinearOp::SaveOutPtr { .. }
            | LinearOp::SlotAddr { .. }
            | LinearOp::LoadFromAddr { .. }
    );

    (def, uses, is_pointer)
}

/// Pretty-print a lockstep result.
pub fn format_result(result: &LockstepResult) -> String {
    let mut out = String::new();

    if let Some(div) = &result.divergence {
        out.push_str(&format!(
            "DIVERGENCE at step {} (DWARF line {})\n",
            div.step, div.dwarf_line
        ));
        out.push_str(&format!("  source: {}\n", div.source_line));
        out.push_str("\n");

        for diff in &div.vreg_diffs {
            let loc_str = match &diff.jit_location {
                VRegLocation::Register(p) => LocationMap::reg_name(*p).to_string(),
                VRegLocation::StackSlot(off) => format!("[sp+{off}]"),
                VRegLocation::Constant(v) => format!("const({v})"),
            };

            let jit_str = diff
                .jit_value
                .map(|v| format!("{v}"))
                .unwrap_or_else(|| "???".to_string());

            let marker = if diff.matches { "  " } else { "!!" };
            out.push_str(&format!(
                "  {marker} v{}: interp={}, jit({})={}\n",
                diff.vreg_index, diff.interpreter_value, loc_str, jit_str
            ));
        }
    } else if result.completed {
        out.push_str(&format!(
            "OK: completed {} steps with no divergence\n",
            result.steps
        ));
    } else {
        out.push_str(&format!(
            "INCOMPLETE: stopped after {} steps (max reached)\n",
            result.steps
        ));
    }

    out
}
