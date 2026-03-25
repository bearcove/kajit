//! Lockstep differential debugger.
//!
//! Drives the CFG-MIR interpreter and a JIT process (via LLDB) simultaneously,
//! comparing vreg values after each CFG-MIR operation and stopping on the first
//! divergence.

use std::collections::HashMap;
use std::path::Path;

use crate::harness::{AllocationMap, VRegLocation};

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
}

#[derive(Debug)]
pub enum DebugError {
    ProcessExited(i32),
    LldbError(String),
    Io(std::io::Error),
}

impl std::fmt::Display for DebugError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DebugError::ProcessExited(code) => write!(f, "process exited with code {code}"),
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
    alloc_map: &AllocationMap,
    listing_lines: &[String],
    debugger: &mut dyn JitDebugger,
    max_steps: usize,
) -> Result<LockstepResult, DebugError> {
    // Read the JIT's base addresses from its registers at the breakpoint.
    // After the aarch64 prologue: x19=cursor, x20=input_end, x21=output_ptr, x22=ctx
    let jit_cursor = debugger.read_register(19)?; // x19
    let jit_input_end = debugger.read_register(20)?; // x20
    let jit_output_ptr = debugger.read_register(21)?; // x21

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

    // Build bidirectional mapping: DWARF line ↔ (block, inst_index, is_term)
    let func = &program.funcs[0];
    // Map (block_id, is_terminator, inst_index_if_not_term) → DWARF line
    // For terminators, inst_index doesn't matter — we key by (block, true, 0)
    let mut op_to_line: std::collections::HashMap<(u32, bool, usize), u32> =
        std::collections::HashMap::new();
    let mut next_line = 1u32;
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        for (inst_idx, _) in block.insts.iter().enumerate() {
            op_to_line.insert((block.id.0, false, inst_idx), next_line);
            next_line += 1;
        }
        // Terminator: key with is_term=true, inst_idx=0 (ignored)
        op_to_line.insert((block.id.0, true, 0), next_line);
        next_line += 1;
    }

    let mut jit_steps = 0;
    // Track which DWARF line LLDB was at before stepping
    let mut prev_dwarf_line = debugger.current_source_line().unwrap_or(1);

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
        let dwarf_line = match debugger.step_to_next_source_line() {
            Ok(line) => line,
            Err(DebugError::ProcessExited(_)) => {
                eprintln!("[lockstep] JIT exited after {jit_steps} steps");
                return Ok(LockstepResult {
                    steps: jit_steps,
                    divergence: None,
                    completed: true,
                });
            }
            Err(e) => return Err(e),
        };
        jit_steps += 1;

        // The JIT was at prev_dwarf_line, stepped to dwarf_line.
        // That means it executed the op at prev_dwarf_line.
        let executed_line = prev_dwarf_line;
        prev_dwarf_line = dwarf_line;

        // Step interpreter to the executed line. Track where it actually goes.
        let mut last_event = None;
        let mut interp_line = 0u32;
        for _ in 0..1000 {
            let event = session
                .step_forward()
                .map_err(|e| DebugError::LldbError(format!("interpreter step: {e}")))?;

            let loc = &event.location_before;
            interp_line = op_to_line
                .get(&(
                    loc.block.0,
                    loc.at_terminator,
                    if loc.at_terminator {
                        0
                    } else {
                        loc.next_inst_index
                    },
                ))
                .copied()
                .unwrap_or(0);

            last_event = Some(event.clone());

            if event.returned || interp_line == executed_line {
                break;
            }
        }

        let Some(event) = last_event else { continue };
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

        if interp_next_line != dwarf_line && !state.returned && dwarf_line != 0 {
            // CONTROL FLOW DIVERGENCE
            let jit_pc = debugger.read_pc()?;

            let exec_source =
                if executed_line > 0 && (executed_line as usize) <= listing_lines.len() {
                    listing_lines[executed_line as usize - 1].clone()
                } else {
                    format!("<line {executed_line}>")
                };
            let jit_target = if dwarf_line > 0 && (dwarf_line as usize) <= listing_lines.len() {
                listing_lines[dwarf_line as usize - 1].clone()
            } else {
                format!("<line {dwarf_line}>")
            };
            let interp_target =
                if interp_next_line > 0 && (interp_next_line as usize) <= listing_lines.len() {
                    listing_lines[interp_next_line as usize - 1].clone()
                } else {
                    format!("<line {interp_next_line}>")
                };

            let mut vreg_diffs = Vec::new();
            // Show the branch condition if it was a terminator
            if let (Some(dst), Some(location)) = (
                def_vreg,
                def_vreg.and_then(|d| alloc_map.locations.get(&(d.index() as u32))),
            ) {
                let sp = debugger.read_sp()?;
                let interp_val = if dst.index() < state.vregs.len() {
                    state.vregs[dst.index()]
                } else {
                    0
                };
                let jit_val = read_vreg_from_jit(debugger, location, sp)?;
                vreg_diffs.push(VRegDiff {
                    vreg_index: dst.index() as u32,
                    interpreter_value: interp_val,
                    jit_value: jit_val,
                    jit_location: location.clone(),
                    matches: jit_val == Some(interp_val),
                });
            }

            let source_line = format!(
                "CONTROL FLOW DIVERGENCE after: {}\n  JIT went to line {} (pc=0x{:x}): {}\n  Interpreter went to line {}: {}",
                exec_source, dwarf_line, jit_pc, jit_target, interp_next_line, interp_target
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

        // Compare the defined vreg (if any)
        if let Some(dst) = def_vreg {
            let dst_idx = dst.index() as u32;
            if let Some(location) = alloc_map.locations.get(&dst_idx) {
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
                        if let Some(use_loc) = alloc_map.locations.get(&use_idx) {
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
                    let source_line =
                        if executed_line > 0 && (executed_line as usize) <= listing_lines.len() {
                            listing_lines[executed_line as usize - 1].clone()
                        } else {
                            format!("<line {executed_line}>")
                        };
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

                let reg_name = match location {
                    VRegLocation::Register(p) => AllocationMap::reg_name(*p),
                    _ => "stk",
                };
                eprintln!(
                    "[lockstep] line {executed_line}: v{}({}) = {} OK",
                    dst_idx, reg_name, interp_value
                );
            }
        }
    }
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
                VRegLocation::Register(p) => AllocationMap::reg_name(*p).to_string(),
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
