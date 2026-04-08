//! Lockstep differential debugger.
//!
//! Drives the CFG-MIR interpreter and a JIT process (via LLDB) simultaneously,
//! comparing vreg values after each CFG-MIR operation and stopping on the first
//! divergence.

use crate::harness::{LocationMap, LocationTracker, VRegLocation};
use kajit_lir::{BinOpKind, LinearOp};
use std::time::{Duration, Instant};

/// Snapshot of JIT register/stack state captured before stepping.
/// Used to compare vreg values for the executed op without ABI clobber interference.
#[derive(Debug, Clone)]
struct RegisterSnapshot {
    sp: u64,
    /// Register values indexed by hw encoding (0..=28 for aarch64).
    regs: std::collections::HashMap<u8, u64>,
    /// Stack memory reads cached by address.
    stack_cache: std::collections::HashMap<u64, u64>,
}

impl RegisterSnapshot {
    /// Capture current register state from the debugger.
    fn capture(debugger: &impl JitDebugger) -> Result<Self, DebugError> {
        let sp = debugger.read_sp()?;
        let mut regs = std::collections::HashMap::new();
        // Capture all GP registers (aarch64: x0-x28, x86_64: 0-15)
        for preg in 0..=28u8 {
            if let Ok(val) = debugger.read_register(preg) {
                regs.insert(preg, val);
            }
        }
        Ok(Self {
            sp,
            regs,
            stack_cache: std::collections::HashMap::new(),
        })
    }

    /// Read a vreg value from this snapshot.
    fn read_vreg(
        &mut self,
        debugger: &impl JitDebugger,
        location: &VRegLocation,
    ) -> Result<Option<u64>, DebugError> {
        match location {
            VRegLocation::Register(preg) => Ok(self.regs.get(preg).copied()),
            VRegLocation::StackSlot(offset) => {
                let addr = self.sp + *offset as u64;
                if let Some(&cached) = self.stack_cache.get(&addr) {
                    return Ok(Some(cached));
                }
                let val = debugger
                    .read_memory(addr, 8)
                    .ok()
                    .map(|b| u64::from_le_bytes(b[..8].try_into().unwrap()));
                if let Some(v) = val {
                    self.stack_cache.insert(addr, v);
                }
                Ok(val)
            }
            VRegLocation::Constant(val) => Ok(Some(*val)),
        }
    }
}

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

/// Tri-state result of comparing a vreg between interpreter and JIT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareResult {
    /// Values match (scalar equality or pointer offset equality).
    Match,
    /// Values diverge.
    Diverged,
    /// Cannot compare (provenance lost, missing pair, etc.).
    Unverified { reason: &'static str },
}

/// Tracks comparison coverage statistics.
#[derive(Debug, Clone, Default)]
pub struct CompareStats {
    pub scalar_matches: u64,
    pub pointer_matches: u64,
    pub unverified_provenance_lost: u64,
    pub unverified_missing_pair: u64,
    pub legacy_skipped: u64,
    pub divergences: u64,
}

impl std::fmt::Display for CompareStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "scalar_match={} ptr_match={} unverified_provenance={} unverified_pair={} legacy_skip={} diverged={}",
            self.scalar_matches,
            self.pointer_matches,
            self.unverified_provenance_lost,
            self.unverified_missing_pair,
            self.legacy_skipped,
            self.divergences
        )
    }
}

/// Paired base addresses for a symbolic pointer: interpreter and JIT sides.
#[derive(Debug, Clone, Copy)]
pub struct PtrPair {
    pub interp_base: u64,
    pub jit_base: u64,
}

/// Compare interpreter tagged value against JIT concrete value using pointer pairs.
fn compare_tagged(
    interp_tagged: kajit_mir::TaggedValue,
    jit_value: Option<u64>,
    ptr_pairs: &std::collections::HashMap<kajit_mir::PtrId, PtrPair>,
) -> CompareResult {
    match interp_tagged {
        kajit_mir::TaggedValue::Scalar(v) => {
            if jit_value == Some(v) {
                CompareResult::Match
            } else {
                CompareResult::Diverged
            }
        }
        kajit_mir::TaggedValue::Pointer {
            id,
            offset: interp_offset,
            ..
        } => {
            if let (Some(jit_val), Some(pair)) = (jit_value, ptr_pairs.get(&id)) {
                let jit_offset = jit_val.wrapping_sub(pair.jit_base);
                if interp_offset == jit_offset {
                    CompareResult::Match
                } else {
                    CompareResult::Diverged
                }
            } else {
                CompareResult::Unverified {
                    reason: "missing JIT value or PtrPair",
                }
            }
        }
        kajit_mir::TaggedValue::UnknownPointer(_) => CompareResult::Unverified {
            reason: "provenance lost",
        },
    }
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

    /// Return a handle that can interrupt the debuggee from another thread.
    /// Returns None if interruption is not supported.
    fn interrupt_handle(&self) -> Option<Box<dyn InterruptHandle>>;
}

/// Thread-safe handle to interrupt a debuggee process.
pub trait InterruptHandle: Send + Sync {
    fn interrupt(&self);
}

/// RAII watchdog: interrupts the debuggee if dropped after 5s.
struct StepWatchdog {
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl StepWatchdog {
    fn new(debugger: &impl JitDebugger) -> Self {
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let thread = if let Some(handle) = debugger.interrupt_handle() {
            let done2 = done.clone();
            Some(std::thread::spawn(move || {
                for _ in 0..50 {
                    std::thread::sleep(Duration::from_millis(100));
                    if done2.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                }
                eprintln!("[lockstep watchdog] 5s timeout — interrupting debuggee");
                handle.interrupt();
            }))
        } else {
            None
        };
        Self { done, thread }
    }
}

impl Drop for StepWatchdog {
    fn drop(&mut self) {
        self.done.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl<T: JitDebugger + ?Sized> JitDebugger for &mut T {
    fn step_to_next_source_line(&mut self) -> Result<u32, DebugError> {
        (**self).step_to_next_source_line()
    }

    fn step_instruction_over(&mut self) -> Result<(), DebugError> {
        (**self).step_instruction_over()
    }

    fn read_register(&self, preg: u8) -> Result<u64, DebugError> {
        (**self).read_register(preg)
    }

    fn read_memory(&self, addr: u64, size: usize) -> Result<Vec<u8>, DebugError> {
        (**self).read_memory(addr, size)
    }

    fn read_sp(&self) -> Result<u64, DebugError> {
        (**self).read_sp()
    }

    fn read_pc(&self) -> Result<u64, DebugError> {
        (**self).read_pc()
    }

    fn has_exited(&self) -> bool {
        (**self).has_exited()
    }

    fn current_source_line(&self) -> Result<u32, DebugError> {
        (**self).current_source_line()
    }

    fn disassemble_around_pc(&self, context: usize) -> Result<String, DebugError> {
        (**self).disassemble_around_pc(context)
    }

    fn interrupt_handle(&self) -> Option<Box<dyn InterruptHandle>> {
        (**self).interrupt_handle()
    }
}

#[derive(Debug)]
pub enum DebugError {
    /// Process exited normally with exit code.
    ProcessExited(i32),
    /// Process killed by signal (SIGSEGV=11, SIGBUS=10, SIGABRT=6, etc.)
    ProcessSignaled(i32),
    /// Wall-clock timeout while stepping (likely infinite loop in JIT code).
    Timeout(String),
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
            DebugError::Timeout(msg) => write!(f, "timeout: {msg}"),
            DebugError::LldbError(msg) => write!(f, "lldb: {msg}"),
            DebugError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

/// Result of a lockstep debugging session.
#[derive(Debug, Clone)]
pub struct LockstepResult {
    /// Total steps executed before stopping.
    pub steps: usize,
    /// The divergence found, if any.
    pub divergence: Option<Divergence>,
    /// Whether the session completed without divergence.
    pub completed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockstepSessionStatus {
    Running,
    Diverged,
    Completed,
}

pub struct LockstepSession<D: JitDebugger> {
    pub cfg_program: kajit_mir::cfg_mir::Program,
    pub location_map: LocationMap,
    pub location_tracker: LocationTracker,
    pub listing_lines: Vec<String>,
    pub interpreter: kajit_mir::DebuggerSession,
    /// Owns the interpreter's memory allocations (cursor, output, ctx).
    /// Must outlive `interpreter`.
    _interp_allocs: crate::context::InterpreterAllocations,
    pub debugger: D,
    pub op_to_line: std::collections::HashMap<(u32, bool, usize), u32>,
    pub prev_dwarf_line: u32,
    pub jit_steps: usize,
    pub verified: std::collections::HashMap<u32, (u64, u32, usize)>,
    pub non_comparable_vregs: std::collections::HashSet<u32>,
    pub ptr_pairs: std::collections::HashMap<kajit_mir::PtrId, PtrPair>,
    pub compare_stats: CompareStats,
    pub status: LockstepSessionStatus,
    pub divergence: Option<Divergence>,
    entry_pc: u64,
    code_line_ranges: Vec<CodeLineRange>,
}

impl<D: JitDebugger> LockstepSession<D> {
    pub fn new(
        program: &kajit_mir::cfg_mir::Program,
        input: &[u8],
        location_map: &LocationMap,
        listing_lines: Vec<String>,
        backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
        entry_offset: usize,
        debugger: D,
    ) -> Result<Self, DebugError> {
        // Build interpreter-side allocations and Arguments.
        // The interpreter gets its own memory, independent from the JIT process.
        let mut interp_allocs =
            crate::context::InterpreterAllocations::new(input, program.funcs[0].output_size);
        let interp_args = interp_allocs.to_arguments(&program.data_arg_layouts);

        // Read JIT register values (for comparison only, NOT passed to interpreter).
        let num_data_args = program.funcs[0].data_args.len();
        let mut jit_register_values = Vec::new();
        for i in 0..num_data_args {
            jit_register_values.push(debugger.read_register(i as u8)?);
        }
        tracing::info!(
            ?jit_register_values,
            "lockstep: JIT register values (for comparison)"
        );

        tracing::info!("lockstep: creating interpreter session");
        let interpreter = kajit_mir::DebuggerSession::new(program, &interp_args)
            .map_err(|e| DebugError::LldbError(format!("interpreter init: {e}")))?;
        tracing::info!("lockstep: interpreter session created");

        let op_to_line = build_op_to_line_map(program, backend_debug_info);
        tracing::info!("lockstep: reading entry PC");
        let entry_pc = debugger.read_pc()?;
        tracing::info!(
            entry_pc = format_args!("0x{:x}", entry_pc),
            "lockstep: got entry PC"
        );
        let code_line_ranges = build_code_line_ranges(backend_debug_info, entry_offset);
        let prev_dwarf_line =
            current_line_from_pc(&debugger, entry_pc, &code_line_ranges).unwrap_or(1);

        // Seed PtrPairs for data_args: pair interpreter PtrIds with JIT register values.
        let mut ptr_pairs = std::collections::HashMap::new();
        for (i, &vreg) in program.funcs[0].data_args.iter().enumerate() {
            let interp_tagged = interpreter.inspect_vreg_tagged(vreg.index());
            if let kajit_mir::TaggedValue::Pointer {
                id,
                concrete: interp_base,
                ..
            } = interp_tagged
            {
                if let Some(&jit_base) = jit_register_values.get(i) {
                    ptr_pairs.insert(
                        id,
                        PtrPair {
                            interp_base,
                            jit_base,
                        },
                    );
                }
            }
        }

        Ok(Self {
            cfg_program: program.clone(),
            location_map: location_map.clone(),
            location_tracker: LocationTracker::new(location_map, program),
            listing_lines,
            interpreter,
            _interp_allocs: interp_allocs,
            debugger,
            op_to_line,
            prev_dwarf_line,
            jit_steps: 0,
            verified: std::collections::HashMap::new(),
            non_comparable_vregs: std::collections::HashSet::new(),
            ptr_pairs,
            compare_stats: CompareStats::default(),
            status: LockstepSessionStatus::Running,
            divergence: None,
            entry_pc,
            code_line_ranges,
        })
    }

    pub fn is_running(&self) -> bool {
        self.status == LockstepSessionStatus::Running
    }

    pub fn current_result(&self) -> LockstepResult {
        LockstepResult {
            steps: self.jit_steps,
            divergence: self.divergence.clone(),
            completed: self.status == LockstepSessionStatus::Completed,
        }
    }

    pub fn current_line_text(&self, line: u32) -> String {
        if line > 0 && (line as usize) <= self.listing_lines.len() {
            self.listing_lines[line as usize - 1].clone()
        } else {
            format!("<line {line}>")
        }
    }

    pub fn current_mapped_line(&self) -> u32 {
        current_line_from_pc(&self.debugger, self.entry_pc, &self.code_line_ranges)
            .unwrap_or(self.prev_dwarf_line)
    }

    fn finish_with_result(&mut self, result: LockstepResult) -> String {
        self.divergence = result.divergence.clone();
        self.status = if result.completed {
            LockstepSessionStatus::Completed
        } else {
            LockstepSessionStatus::Diverged
        };
        format_result(&result)
    }

    pub fn step_forward(&mut self) -> Result<String, DebugError> {
        if self.status != LockstepSessionStatus::Running {
            return Ok(format_result(&self.current_result()));
        }

        if self.debugger.has_exited() {
            let result = handle_jit_exit(
                &mut self.interpreter,
                &self.op_to_line,
                &self.listing_lines,
                self.jit_steps,
                self.prev_dwarf_line,
                "already exited",
            )?;
            return Ok(self.finish_with_result(result));
        }

        // Spawn a watchdog that interrupts the debuggee if step_forward takes >5s.
        let _watchdog = StepWatchdog::new(&self.debugger);

        // Capture register state BEFORE stepping — the step may clobber registers
        // (e.g. ABI arg setup for call_intrinsic overwrites argument registers).
        let mut pre_step_snapshot = RegisterSnapshot::capture(&self.debugger)?;

        let dwarf_line = match step_to_next_mapped_line(
            &mut self.debugger,
            self.entry_pc,
            &self.code_line_ranges,
            self.prev_dwarf_line,
        ) {
            Ok(line) => line,
            Err(DebugError::ProcessExited(code)) => {
                let result = handle_jit_exit(
                    &mut self.interpreter,
                    &self.op_to_line,
                    &self.listing_lines,
                    self.jit_steps,
                    self.prev_dwarf_line,
                    &format!("exited with code {code}"),
                )?;
                return Ok(self.finish_with_result(result));
            }
            Err(DebugError::ProcessSignaled(sig)) => {
                let sig_name = match sig {
                    6 => "SIGABRT",
                    10 => "SIGBUS",
                    11 => "SIGSEGV",
                    _ => "signal",
                };
                let result = handle_jit_exit(
                    &mut self.interpreter,
                    &self.op_to_line,
                    &self.listing_lines,
                    self.jit_steps,
                    self.prev_dwarf_line,
                    &format!("killed by {sig_name} (signal {sig})"),
                )?;
                return Ok(self.finish_with_result(result));
            }
            Err(DebugError::Timeout(msg)) => {
                let result = handle_jit_exit(
                    &mut self.interpreter,
                    &self.op_to_line,
                    &self.listing_lines,
                    self.jit_steps,
                    self.prev_dwarf_line,
                    &format!("timeout: {msg}"),
                )?;
                return Ok(self.finish_with_result(result));
            }
            Err(e) => return Err(e),
        };

        self.jit_steps += 1;
        let executed_line = self.prev_dwarf_line;
        self.prev_dwarf_line = dwarf_line;
        let func = &self.cfg_program.funcs[0];

        let mut last_event = None;
        let mut executed_state = None;
        let mut tracker = self.location_tracker.clone();
        let mut non_comparable = self.non_comparable_vregs.clone();
        let mut executed_tracker = None;
        let mut executed_non_comparable = None;
        let mut synced = false;
        let interp_deadline = Instant::now() + STEP_TIMEOUT;
        for interp_step in 0..500u32 {
            if interp_step % 50 == 49 && Instant::now() > interp_deadline {
                return Err(DebugError::Timeout(format!(
                    "interpreter sync loop timed out after {interp_step} steps \
                     (trying to sync from line {executed_line} to line {dwarf_line})"
                )));
            }
            let pre_loc = self.interpreter.state().location;
            let pre_line = loc_to_line(&self.op_to_line, &pre_loc);

            if pre_line == executed_line && last_event.is_none() {
                let event = self
                    .interpreter
                    .step_forward()
                    .map_err(|e| DebugError::LldbError(format!("interpreter step: {e}")))?;
                let post_loc = self.interpreter.state().location;
                tracker.observe_step(
                    &self.location_map,
                    func,
                    executed_line,
                    &event.location_before,
                    &post_loc,
                );
                observe_non_comparable_step(
                    &mut non_comparable,
                    func,
                    &event.location_before,
                    &post_loc,
                );
                executed_state = Some(self.interpreter.state());
                executed_tracker = Some(tracker.clone());
                executed_non_comparable = Some(non_comparable.clone());
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

            let event = self
                .interpreter
                .step_forward()
                .map_err(|e| DebugError::LldbError(format!("interpreter step: {e}")))?;
            let post_loc = self.interpreter.state().location;

            let ev_line = loc_to_line(&self.op_to_line, &event.location_before);
            tracker.observe_step(
                &self.location_map,
                func,
                ev_line,
                &event.location_before,
                &post_loc,
            );
            observe_non_comparable_step(
                &mut non_comparable,
                func,
                &event.location_before,
                &post_loc,
            );
            if ev_line == executed_line && last_event.is_none() {
                executed_state = Some(self.interpreter.state());
                executed_tracker = Some(tracker.clone());
                executed_non_comparable = Some(non_comparable.clone());
                last_event = Some(event.clone());
            }

            if event.returned {
                break;
            }
        }

        if !synced && !self.interpreter.state().returned {
            let cur_loc = self.interpreter.state().location;
            let cur_line = loc_to_line(&self.op_to_line, &cur_loc);
            let disasm = self.debugger.disassemble_around_pc(4).unwrap_or_default();
            let source_line = format!(
                "\
SYNC FAILURE at JIT step {}

  JIT executed line {}: {}
  JIT now at line {}: {}
  interpreter stuck at line {} (b{} inst[{}] term={}): {}

  machine code at JIT position:
{}",
                self.jit_steps,
                executed_line,
                self.current_line_text(executed_line),
                dwarf_line,
                self.current_line_text(dwarf_line),
                cur_line,
                cur_loc.block.index(),
                cur_loc.next_inst_index,
                cur_loc.at_terminator,
                self.current_line_text(cur_line),
                disasm,
            );
            let result = LockstepResult {
                steps: self.jit_steps,
                divergence: Some(Divergence {
                    step: self.jit_steps,
                    dwarf_line: executed_line,
                    source_line,
                    vreg_diffs: Vec::new(),
                }),
                completed: false,
            };
            return Ok(self.finish_with_result(result));
        }

        let Some(event) = last_event else {
            let cur_loc = self.interpreter.state().location;
            let cur_line = loc_to_line(&self.op_to_line, &cur_loc);
            let disasm = self.debugger.disassemble_around_pc(4).unwrap_or_default();
            let source_line = format!(
                "\
LOCKSTEP DESYNC at JIT step {}

  JIT executed line {}: {}
  JIT now at line {}: {}
  interpreter at line {} (b{} inst[{}] term={}): {}

  The interpreter could not find line {}.

  machine code at JIT position:
{}",
                self.jit_steps,
                executed_line,
                self.current_line_text(executed_line),
                dwarf_line,
                self.current_line_text(dwarf_line),
                cur_line,
                cur_loc.block.index(),
                cur_loc.next_inst_index,
                cur_loc.at_terminator,
                self.current_line_text(cur_line),
                executed_line,
                disasm,
            );
            let result = LockstepResult {
                steps: self.jit_steps,
                divergence: Some(Divergence {
                    step: self.jit_steps,
                    dwarf_line: executed_line,
                    source_line,
                    vreg_diffs: Vec::new(),
                }),
                completed: false,
            };
            return Ok(self.finish_with_result(result));
        };

        self.location_tracker = tracker;
        self.non_comparable_vregs = non_comparable;

        let state = executed_state.unwrap_or_else(|| self.interpreter.state());
        let loc = &event.location_before;
        let (def_vreg, use_vregs, _) = op_def_uses_and_kind(func, loc);
        let compare_tracker = executed_tracker.as_ref().unwrap_or(&self.location_tracker);
        let compare_non_comparable = executed_non_comparable
            .as_ref()
            .unwrap_or(&self.non_comparable_vregs);

        let interp_next_loc = self.interpreter.state().location;
        let interp_next_line = self
            .op_to_line
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
            let jit_pc = self.debugger.read_pc()?;
            let disasm = self.debugger.disassemble_around_pc(4).unwrap_or_default();
            let term_info = {
                let block = &func.blocks[loc.block.index()];
                let term = &func.terms[block.term.index()];
                let mut info = format!("  terminator: {:?}\n", term);
                for &edge_id in &block.succs {
                    let edge = &func.edges[edge_id.index()];
                    let target_line = self
                        .op_to_line
                        .get(&(edge.to.0, false, 0))
                        .or_else(|| self.op_to_line.get(&(edge.to.0, true, 0)))
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
                def_vreg.and_then(|d| {
                    compare_tracker.location_for(&self.location_map, d.index() as u32)
                }),
            ) {
                let iv = if dst.index() < state.vregs.len() {
                    state.vregs[dst.index()]
                } else {
                    0
                };
                // Def vreg: read AFTER step (the instruction just wrote it)
                let sp = self.debugger.read_sp()?;
                let jv = read_vreg_live(&self.debugger, &location, sp)?;
                vreg_diffs.push(VRegDiff {
                    vreg_index: dst.index() as u32,
                    interpreter_value: iv,
                    jit_value: jv,
                    jit_location: location,
                    matches: jv == Some(iv),
                });
            }

            let source_line = format!(
                "\
CONTROL FLOW DIVERGENCE at step {}

  last op (line {}): {}
{}  JIT went to:         line {} (pc=0x{:x}): {}
  interpreter went to: line {}: {}

  machine code:
{}",
                self.jit_steps,
                executed_line,
                self.current_line_text(executed_line),
                term_info,
                dwarf_line,
                jit_pc,
                self.current_line_text(dwarf_line),
                interp_next_line,
                self.current_line_text(interp_next_line),
                disasm,
            );
            let result = LockstepResult {
                steps: self.jit_steps,
                divergence: Some(Divergence {
                    step: self.jit_steps,
                    dwarf_line: executed_line,
                    source_line,
                    vreg_diffs,
                }),
                completed: false,
            };
            return Ok(self.finish_with_result(result));
        }

        if should_skip_fused_compare_value_check(func, loc, def_vreg)
            || should_skip_branch_predicate_value_check(func, loc)
            || should_skip_process_local_intrinsic_return_check(func, loc)
            || should_skip_non_comparable_value_check(def_vreg, &use_vregs, compare_non_comparable)
        {
            self.compare_stats.legacy_skipped += 1;
            return Ok(format!(
                "step {}: line {} skipped (non-comparable backend artifact)",
                self.jit_steps, executed_line
            ));
        }

        let compare_vregs: Vec<kajit_ir::VReg> = if let Some(dst) = def_vreg {
            vec![dst]
        } else if !use_vregs.is_empty() {
            use_vregs.clone()
        } else {
            Vec::new()
        };

        if let Some(&dst) = compare_vregs.first() {
            let dst_idx = dst.index() as u32;
            let location = compare_tracker.location_for(&self.location_map, dst_idx);
            if let Some(location) = location {
                let interp_tagged = if dst.index() < state.tagged_vregs.len() {
                    state.tagged_vregs[dst.index()]
                } else {
                    kajit_mir::TaggedValue::Scalar(0)
                };
                let interp_value = interp_tagged.concrete();

                // Def vreg: read AFTER step (the instruction just wrote it)
                // Use vreg (no def): read from pre-step snapshot (may be clobbered by next op's ABI setup)
                let jit_value = if def_vreg.is_some() {
                    let post_sp = self.debugger.read_sp()?;
                    read_vreg_live(&self.debugger, &location, post_sp)?
                } else {
                    pre_step_snapshot.read_vreg(&self.debugger, &location)?
                };

                // Record PtrPair for pointer birth ops
                if let kajit_mir::TaggedValue::Pointer {
                    id,
                    concrete: interp_base,
                    offset: 0,
                    ..
                } = interp_tagged
                {
                    if let Some(jit_val) = jit_value {
                        if !self.ptr_pairs.contains_key(&id) {
                            self.ptr_pairs.insert(
                                id,
                                PtrPair {
                                    interp_base,
                                    jit_base: jit_val,
                                },
                            );
                        }
                    }
                }

                let cmp = compare_tagged(interp_tagged, jit_value, &self.ptr_pairs);
                match cmp {
                    CompareResult::Match => {
                        if interp_tagged.is_pointer() {
                            self.compare_stats.pointer_matches += 1;
                        } else {
                            self.compare_stats.scalar_matches += 1;
                        }
                    }
                    CompareResult::Unverified { reason } => {
                        if matches!(interp_tagged, kajit_mir::TaggedValue::UnknownPointer(_)) {
                            self.compare_stats.unverified_provenance_lost += 1;
                        } else {
                            self.compare_stats.unverified_missing_pair += 1;
                        }
                        let reg_name = match &location {
                            VRegLocation::Register(p) => LocationMap::reg_name(*p),
                            _ => "stk",
                        };
                        return Ok(format!(
                            "step {}: line {} UNVERIFIED v{}({}) — {}",
                            self.jit_steps, executed_line, dst_idx, reg_name, reason
                        ));
                    }
                    CompareResult::Diverged => {
                        self.compare_stats.divergences += 1;
                    }
                }

                if cmp == CompareResult::Diverged {
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
                            compare_tracker.location_for(&self.location_map, use_idx)
                        {
                            let use_interp = if use_vreg.index() < state.vregs.len() {
                                state.vregs[use_vreg.index()]
                            } else {
                                0
                            };
                            let use_jit = pre_step_snapshot.read_vreg(&self.debugger, &use_loc)?;
                            vreg_diffs.push(VRegDiff {
                                vreg_index: use_idx,
                                interpreter_value: use_interp,
                                jit_value: use_jit,
                                jit_location: use_loc,
                                matches: use_jit == Some(use_interp),
                            });
                        }
                    }
                    let disasm = self.debugger.disassemble_around_pc(4).unwrap_or_default();
                    let mut diag = String::new();
                    diag.push_str(&format!(
                        "VALUE DIVERGENCE at step {}, line {}\n\n",
                        self.jit_steps, executed_line
                    ));
                    diag.push_str(&format!(
                        "  op: {}\n",
                        self.current_line_text(executed_line)
                    ));
                    let reg_name = match location {
                        VRegLocation::Register(p) => LocationMap::reg_name(p),
                        _ => "stk",
                    };
                    diag.push_str(&format!(
                        "\n  v{} ({}): interpreter={}, jit={}\n",
                        dst_idx,
                        reg_name,
                        interp_value,
                        jit_value.map(|v| v.to_string()).unwrap_or("???".into())
                    ));
                    if let Some(&(last_val, last_line, last_step)) = self.verified.get(&dst_idx) {
                        diag.push_str(&format!(
                            "  v{} last verified: {} at line {} (step {})\n",
                            dst_idx, last_val, last_line, last_step
                        ));
                    } else {
                        diag.push_str(&format!("  v{} was NEVER verified before\n", dst_idx));
                    }

                    if let VRegLocation::Register(preg) = location {
                        let sharing: Vec<_> = self
                            .location_map
                            .static_locations
                            .iter()
                            .filter(|(k, v)| {
                                **k != dst_idx
                                    && matches!(v, VRegLocation::Register(p) if *p == preg)
                            })
                            .map(|(k, _)| *k)
                            .collect();
                        if !sharing.is_empty() {
                            diag.push_str(&format!(
                                "\n  other vregs sharing {}: ",
                                LocationMap::reg_name(preg)
                            ));
                            for (i, v) in sharing.iter().enumerate() {
                                if i > 0 {
                                    diag.push_str(", ");
                                }
                                if let Some(&(val, line, _step)) = self.verified.get(v) {
                                    diag.push_str(&format!("v{}(={} @line{})", v, val, line));
                                } else {
                                    diag.push_str(&format!("v{}", v));
                                }
                            }
                            diag.push('\n');

                            if let Some(jv) = jit_value {
                                for &v in &sharing {
                                    if let Some(&(val, line, _)) = self.verified.get(&v)
                                        && val == jv
                                    {
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

                    let result = LockstepResult {
                        steps: self.jit_steps,
                        divergence: Some(Divergence {
                            step: self.jit_steps,
                            dwarf_line: executed_line,
                            source_line: diag,
                            vreg_diffs,
                        }),
                        completed: false,
                    };
                    return Ok(self.finish_with_result(result));
                }

                self.verified
                    .insert(dst_idx, (interp_value, executed_line, self.jit_steps));

                let reg_name = match location {
                    VRegLocation::Register(p) => LocationMap::reg_name(p),
                    _ => "stk",
                };
                return Ok(format!(
                    "step {}: line {} OK, v{}({}) = {}",
                    self.jit_steps, executed_line, dst_idx, reg_name, interp_value
                ));
            } else if self.location_map.call_lines.contains(&executed_line) {
                return Ok(format!(
                    "step {}: line {} skipped (call-clobbered location)",
                    self.jit_steps, executed_line
                ));
            }
        }

        Ok(format!(
            "step {}: line {} executed, no comparable vreg",
            self.jit_steps, executed_line
        ))
    }
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
    let mut session = LockstepSession::new(
        program,
        input,
        location_map,
        listing_lines.to_vec(),
        backend_debug_info,
        entry_offset,
        debugger,
    )?;

    loop {
        if session.jit_steps >= max_steps {
            eprintln!("[lockstep] max steps ({max_steps}) reached");
            return Ok(LockstepResult {
                steps: session.jit_steps,
                divergence: None,
                completed: false,
            });
        }

        let msg = session.step_forward()?;
        if session.is_running() {
            eprintln!("[lockstep] {msg}");
            continue;
        }

        return Ok(session.current_result());
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

fn build_op_to_line_map(
    program: &kajit_mir::cfg_mir::Program,
    backend_debug_info: Option<&crate::ir_backend::BackendDebugInfo>,
) -> std::collections::HashMap<(u32, bool, usize), u32> {
    let func = &program.funcs[0];
    let lambda_id = func.lambda_id.index() as u32;

    let mut op_locations =
        std::collections::HashMap::<kajit_mir::cfg_mir::OpId, (u32, bool, usize)>::new();
    let mut fallback = std::collections::HashMap::<(u32, bool, usize), u32>::new();
    let mut next_line = 1u32;
    for block in func.live_blocks() {
        for (inst_idx, inst_id) in block.insts.iter().enumerate() {
            let key = (block.id.0, false, inst_idx);
            op_locations.insert(kajit_mir::cfg_mir::OpId::Inst(*inst_id), key);
            fallback.insert(key, next_line);
            next_line += 1;
        }
        let term_key = (block.id.0, true, 0);
        op_locations.insert(kajit_mir::cfg_mir::OpId::Term(block.term), term_key);
        fallback.insert(term_key, next_line);
        next_line += 1;
    }

    let Some(backend_debug_info) = backend_debug_info else {
        return fallback;
    };

    let mut mapped = fallback.clone();
    for op in &backend_debug_info.op_infos {
        if op.lambda_id != lambda_id {
            continue;
        }
        if let Some(&key) = op_locations.get(&op.op_id) {
            mapped.insert(key, op.line);
        }
    }
    mapped
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
    if !code_line_ranges.is_empty() {
        return Ok(0);
    }
    Ok(debugger.current_source_line().unwrap_or(0))
}

/// Maximum wall-clock time for a single `step_to_next_mapped_line` call.
/// If the JIT code is stuck in an infinite loop (e.g. inside a call_intrinsic),
/// `step_instruction_over` blocks until the call returns — this deadline
/// ensures we don't hang the MCP server forever.
const STEP_TIMEOUT: Duration = Duration::from_secs(5);

fn step_to_next_mapped_line(
    debugger: &mut dyn JitDebugger,
    entry_pc: u64,
    code_line_ranges: &[CodeLineRange],
    start_line: u32,
) -> Result<u32, DebugError> {
    let deadline = Instant::now() + STEP_TIMEOUT;
    for i in 0..4096u32 {
        debugger.step_instruction_over()?;
        if Instant::now() > deadline {
            let pc = debugger.read_pc().unwrap_or(0);
            let disasm = debugger
                .disassemble_around_pc(8)
                .unwrap_or_else(|_| "(disassembly unavailable)".to_string());
            return Err(DebugError::Timeout(format!(
                "timed out after {i} instruction steps \
                 (line {start_line}, pc=0x{pc:x}). JIT code likely in infinite loop.\n\n\
                 Disassembly around stuck PC:\n{disasm}"
            )));
        }
        let line = current_line_from_pc(debugger, entry_pc, code_line_ranges)?;
        if line != 0 && line != start_line {
            return Ok(line);
        }
    }

    Err(DebugError::LldbError(format!(
        "instruction stepping did not leave mapped line {start_line}"
    )))
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

fn should_skip_non_comparable_value_check(
    def_vreg: Option<kajit_ir::VReg>,
    use_vregs: &[kajit_ir::VReg],
    non_comparable_vregs: &std::collections::HashSet<u32>,
) -> bool {
    def_vreg.is_some_and(|dst| non_comparable_vregs.contains(&(dst.index() as u32)))
        || use_vregs
            .iter()
            .any(|vreg| non_comparable_vregs.contains(&(vreg.index() as u32)))
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

fn observe_non_comparable_step(
    non_comparable_vregs: &mut std::collections::HashSet<u32>,
    func: &kajit_mir::cfg_mir::Function,
    loc_before: &kajit_mir::ProgramLocation,
    loc_after: &kajit_mir::ProgramLocation,
) {
    if loc_before.at_terminator {
        if let Some(edge) = chosen_edge_for_non_comparable(func, loc_before, loc_after) {
            for arg in &edge.args {
                let src_idx = arg.source.index() as u32;
                let dst_idx = arg.target.index() as u32;
                if non_comparable_vregs.contains(&src_idx) {
                    non_comparable_vregs.insert(dst_idx);
                } else {
                    non_comparable_vregs.remove(&dst_idx);
                }
            }
        }
        return;
    }

    let Some(block) = func.blocks.get(loc_before.block.index()) else {
        return;
    };
    let Some(&inst_id) = block.insts.get(loc_before.next_inst_index) else {
        return;
    };
    let Some(inst) = func.insts.get(inst_id.index()) else {
        return;
    };
    let Some(def_vreg) = op_def_vreg_at(func, loc_before) else {
        return;
    };
    let def_idx = def_vreg.index() as u32;
    if op_produces_non_comparable_value(&inst.op, non_comparable_vregs) {
        non_comparable_vregs.insert(def_idx);
    } else {
        non_comparable_vregs.remove(&def_idx);
    }
}

fn op_produces_non_comparable_value(
    op: &LinearOp,
    non_comparable_vregs: &std::collections::HashSet<u32>,
) -> bool {
    match op {
        LinearOp::SlotAddr { .. } | LinearOp::ExternAddr { .. } | LinearOp::StackAlloc { .. } => {
            true
        }
        LinearOp::Copy { src, .. } => non_comparable_vregs.contains(&(src.index() as u32)),
        LinearOp::BinOp {
            op: BinOpKind::Add | BinOpKind::Sub,
            lhs,
            rhs,
            ..
        } => {
            non_comparable_vregs.contains(&(lhs.index() as u32))
                || non_comparable_vregs.contains(&(rhs.index() as u32))
        }
        LinearOp::CallIntrinsic { func, dst, .. } if dst.is_some() => {
            is_process_local_pointer_intrinsic(*func)
        }
        LinearOp::CallPure { func, .. } | LinearOp::CallEffect { func, .. } => {
            is_process_local_pointer_intrinsic(*func)
        }
        _ => false,
    }
}

fn chosen_edge_for_non_comparable<'a>(
    func: &'a kajit_mir::cfg_mir::Function,
    loc_before: &kajit_mir::ProgramLocation,
    loc_after: &kajit_mir::ProgramLocation,
) -> Option<&'a kajit_mir::cfg_mir::Edge> {
    let block = func.blocks.get(loc_before.block.index())?;
    let term = func.terms.get(block.term.index())?;
    let edge_id = match term {
        kajit_mir::cfg_mir::Terminator::Branch { edge } => Some(*edge),
        kajit_mir::cfg_mir::Terminator::BranchIf {
            taken, fallthrough, ..
        }
        | kajit_mir::cfg_mir::Terminator::BranchIfZero {
            taken, fallthrough, ..
        } => {
            let taken_edge = func.edges.get(taken.index())?;
            if taken_edge.to == loc_after.block {
                Some(*taken)
            } else {
                Some(*fallthrough)
            }
        }
        kajit_mir::cfg_mir::Terminator::JumpTable {
            targets, default, ..
        } => targets
            .iter()
            .copied()
            .find(|edge_id| func.edges[edge_id.index()].to == loc_after.block)
            .or(Some(*default).filter(|edge_id| func.edges[edge_id.index()].to == loc_after.block)),
        _ => None,
    }?;
    func.edges.get(edge_id.index())
}

fn op_def_vreg_at(
    func: &kajit_mir::cfg_mir::Function,
    loc: &kajit_mir::ProgramLocation,
) -> Option<kajit_ir::VReg> {
    use kajit_mir::cfg_mir::OperandKind;

    if loc.at_terminator {
        return None;
    }
    let block = func.blocks.get(loc.block.index())?;
    let inst_id = *block.insts.get(loc.next_inst_index)?;
    let inst = func.insts.get(inst_id.index())?;
    inst.operands
        .iter()
        .find(|operand| operand.kind == OperandKind::Def)
        .map(|operand| operand.vreg)
}

/// Look up the DWARF line for an interpreter ProgramLocation.
pub fn loc_to_line(
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

/// Read a vreg's current (live) value from the JIT process.
fn read_vreg_live(
    debugger: &impl JitDebugger,
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
pub fn op_def_uses_and_kind(
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
        LinearOp::SlotAddr { .. } | LinearOp::StackAlloc { .. } | LinearOp::LoadFromAddr { .. }
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
        out.push('\n');

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
