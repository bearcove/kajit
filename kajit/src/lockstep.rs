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
    // Set up interpreter
    let mut session = kajit_mir::DebuggerSession::new(program, input)
        .map_err(|e| DebugError::LldbError(format!("interpreter init: {e}")))?;

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

        // Step interpreter one op
        let _event = session
            .step_forward()
            .map_err(|e| DebugError::LldbError(format!("interpreter step: {e}")))?;

        let state = session.state();
        if state.returned {
            eprintln!("[lockstep] interpreter returned after {steps} steps");
            return Ok(LockstepResult {
                steps,
                divergence: None,
                completed: true,
            });
        }

        // Step JIT to next source line
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

        // Compare vreg values
        let sp = debugger.read_sp()?;
        let mut vreg_diffs = Vec::new();
        let mut has_divergence = false;

        // Check all vregs that have allocations
        for (&vreg_idx, location) in &alloc_map.locations {
            let vreg_idx_usize = vreg_idx as usize;

            // Get interpreter value (if this vreg is live)
            let interp_value = if vreg_idx_usize < state.vregs.len() {
                state.vregs[vreg_idx_usize]
            } else {
                continue; // vreg not in interpreter state
            };

            // Get JIT value from physical location
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
            if !matches && interp_value != 0 {
                // Only report divergences for non-zero interpreter values
                // (zero vregs are often dead/unused)
                has_divergence = true;
            }

            if !matches {
                vreg_diffs.push(VRegDiff {
                    vreg_index: vreg_idx,
                    interpreter_value: interp_value,
                    jit_value,
                    jit_location: location.clone(),
                    matches,
                });
            }
        }

        steps += 1;

        if has_divergence {
            // Sort diffs by vreg index for readability
            vreg_diffs.sort_by_key(|d| d.vreg_index);

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

        // Progress report every 50 steps
        if steps % 50 == 0 {
            eprintln!("[lockstep] step {steps}, line {dwarf_line}, no divergence yet");
        }
    }
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
