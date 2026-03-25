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

    let mut steps = 0;

    loop {
        if steps >= max_steps {
            eprintln!("[lockstep] max steps ({max_steps}) reached");
            return Ok(LockstepResult {
                steps,
                divergence: None,
                completed: false,
            });
        }

        if debugger.has_exited() {
            return Ok(LockstepResult {
                steps,
                divergence: None,
                completed: true,
            });
        }

        // Step BOTH: JIT first (executes the current op's machine code),
        // then interpreter (executes the same op in the ideal model).
        // After both step, they should agree on the op's output.

        // Step JIT to next source line (executes current op's machine code)
        let dwarf_line = match debugger.step_to_next_source_line() {
            Ok(line) => line,
            Err(DebugError::ProcessExited(_)) => {
                eprintln!("[lockstep] JIT process exited after {steps} steps");
                return Ok(LockstepResult {
                    steps,
                    divergence: None,
                    completed: true,
                });
            }
            Err(e) => return Err(e),
        };

        // Step interpreter one op — capture which op executed
        let event = session
            .step_forward()
            .map_err(|e| DebugError::LldbError(format!("interpreter step: {e}")))?;

        let state = session.state();
        if state.returned || event.returned {
            eprintln!("[lockstep] interpreter returned after {steps} steps");
            return Ok(LockstepResult {
                steps,
                divergence: None,
                completed: true,
            });
        }

        // Find which vregs the executed op defines and uses
        let func = &program.funcs[0]; // single function
        let (def_vreg, use_vregs, _) = op_def_uses_and_kind(func, &event.location_before);

        steps += 1;

        // Compare only the defined vreg (the output of this op)
        let Some(dst) = def_vreg else {
            // Terminators and side-effect-only ops don't define a vreg — skip comparison
            if steps % 50 == 0 {
                eprintln!("[lockstep] step {steps}, line {dwarf_line}");
            }
            continue;
        };

        let dst_idx = dst.index() as u32;
        let Some(location) = alloc_map.locations.get(&dst_idx) else {
            continue; // vreg not allocated (dead?)
        };

        let interp_value = if (dst.index()) < state.vregs.len() {
            state.vregs[dst.index()]
        } else {
            continue;
        };

        let sp = debugger.read_sp()?;
        let jit_value = match location {
            VRegLocation::Register(preg) => debugger.read_register(*preg).ok(),
            VRegLocation::StackSlot(offset) => {
                let addr = sp + *offset as u64;
                debugger
                    .read_memory(addr, 8)
                    .ok()
                    .map(|bytes| u64::from_le_bytes(bytes[..8].try_into().unwrap()))
            }
            VRegLocation::Constant(val) => Some(*val),
        };

        let matches = jit_value == Some(interp_value);

        if !matches {
            // Build context: also read the use vregs
            let mut vreg_diffs = Vec::new();

            // The diverging def vreg
            vreg_diffs.push(VRegDiff {
                vreg_index: dst_idx,
                interpreter_value: interp_value,
                jit_value,
                jit_location: location.clone(),
                matches: false,
            });

            // Context: show the use (input) vregs too
            for use_vreg in &use_vregs {
                let use_idx = use_vreg.index() as u32;
                if let Some(use_loc) = alloc_map.locations.get(&use_idx) {
                    let use_interp = if use_vreg.index() < state.vregs.len() {
                        state.vregs[use_vreg.index()]
                    } else {
                        0
                    };
                    let use_jit = match use_loc {
                        VRegLocation::Register(p) => debugger.read_register(*p).ok(),
                        VRegLocation::StackSlot(off) => {
                            let addr = sp + *off as u64;
                            debugger
                                .read_memory(addr, 8)
                                .ok()
                                .map(|b| u64::from_le_bytes(b[..8].try_into().unwrap()))
                        }
                        VRegLocation::Constant(v) => Some(*v),
                    };
                    vreg_diffs.push(VRegDiff {
                        vreg_index: use_idx,
                        interpreter_value: use_interp,
                        jit_value: use_jit,
                        jit_location: use_loc.clone(),
                        matches: use_jit == Some(use_interp),
                    });
                }
            }

            let source_line = if dwarf_line > 0 && (dwarf_line as usize) <= listing_lines.len() {
                listing_lines[dwarf_line as usize - 1].clone()
            } else {
                format!("<line {dwarf_line}>")
            };

            return Ok(LockstepResult {
                steps,
                divergence: Some(Divergence {
                    step: steps,
                    dwarf_line,
                    source_line,
                    vreg_diffs,
                }),
                completed: false,
            });
        }

        // Progress report
        let loc_str = match &event.location_before.at_terminator {
            true => format!("b{} term", event.location_before.block.index()),
            false => format!(
                "b{} inst[{}]",
                event.location_before.block.index(),
                event.location_before.next_inst_index
            ),
        };
        let loc_name = AllocationMap::reg_name(
            alloc_map
                .locations
                .get(&dst_idx)
                .map(|l| match l {
                    VRegLocation::Register(p) => *p,
                    _ => 255,
                })
                .unwrap_or(255),
        );
        eprintln!(
            "[lockstep] step {steps}: {loc_str} line {dwarf_line} v{dst_idx}({loc_name}) = {interp_value} OK"
        );
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
