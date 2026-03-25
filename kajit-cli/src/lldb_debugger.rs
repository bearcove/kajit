//! LLDB implementation of the JitDebugger trait for lockstep debugging.

use kajit::lockstep::{DebugError, JitDebugger};
use lldb::*;
use std::path::Path;

/// LLDB-based JIT debugger that drives a standalone harness process.
#[allow(dead_code)]
pub struct LldbJitDebugger {
    debugger: SBDebugger,
    target: SBTarget,
    process: SBProcess,
    exited: bool,
}

impl LldbJitDebugger {
    /// Launch a harness executable under LLDB, stopping at CFG-MIR line 1.
    /// The dSYM bundle must be next to the executable (auto-discovered by LLDB).
    pub fn launch(harness_path: &str, input_hex: &str) -> Result<Self, DebugError> {
        SBDebugger::initialize();

        let debugger = SBDebugger::create(false);
        debugger.set_asynchronous(false); // synchronous mode

        let target = debugger.create_target_simple(harness_path).ok_or_else(|| {
            DebugError::LldbError(format!("failed to create target for {harness_path}"))
        })?;

        // Set breakpoint at first CFG-MIR source line (dSYM auto-discovered)
        let listing_name = Path::new(harness_path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        let listing_file = format!("{listing_name}.cfg-mir");
        let bp = target.breakpoint_create_by_location(&listing_file, 1);
        if !bp.is_valid() {
            // Fallback: breakpoint by function name
            eprintln!("[lldb] warning: source breakpoint failed, using function name");
            let bp = target.breakpoint_create_by_name("kajit_decode");
            if !bp.is_valid() {
                return Err(DebugError::LldbError("failed to set breakpoint".into()));
            }
        }

        // Launch
        let launch_info = SBLaunchInfo::new();
        launch_info.set_arguments([input_hex].into_iter(), false);

        let process = target
            .launch(launch_info)
            .map_err(|e| DebugError::LldbError(format!("launch failed: {e}")))?;

        if !process.is_stopped() {
            return Err(DebugError::LldbError("process didn't stop".into()));
        }

        let thread = process.selected_thread();
        let frame = thread.selected_frame();
        let line = frame.line_entry().map(|le| le.line()).unwrap_or(0);

        eprintln!(
            "[lldb] stopped at pc=0x{:x}, line={}, reason={:?}",
            frame.pc(),
            line,
            thread.stop_reason()
        );

        Ok(Self {
            debugger,
            target,
            process,
            exited: false,
        })
    }
}

impl JitDebugger for LldbJitDebugger {
    fn step_to_next_source_line(&mut self) -> Result<u32, DebugError> {
        if self.exited {
            return Err(DebugError::ProcessExited(0));
        }

        let thread = self.process.selected_thread();

        // Source-level step over — advances one CFG-MIR operation
        thread
            .step_over(RunMode::OnlyThisThread)
            .map_err(|e| DebugError::LldbError(format!("step_over: {e}")))?;

        if !self.process.is_stopped() {
            self.exited = true;
            return Err(DebugError::ProcessExited(self.process.exit_status()));
        }

        // Check for signal (SIGSEGV, SIGBUS, SIGABRT, etc.)
        let stop_reason = thread.stop_reason();
        if stop_reason == lldb::StopReason::Signal || stop_reason == lldb::StopReason::Exception {
            self.exited = true;
            // On macOS, Mach exceptions map to Unix signals. The exit status
            // encodes the signal number when the process is killed.
            let status = self.process.exit_status();
            // If process is still alive (stopped on signal), use stop info
            let sig = if self.process.is_alive() {
                // LLDB exposes signal via stop description; approximate from exception type
                match stop_reason {
                    lldb::StopReason::Signal => status,
                    lldb::StopReason::Exception => 11, // SIGSEGV is most common
                    _ => status,
                }
            } else {
                status
            };
            return Err(DebugError::ProcessSignaled(sig));
        }

        // Check if we stepped out of the function
        let frame = thread.selected_frame();
        let line = frame.line_entry().map(|le| le.line()).unwrap_or(0);

        if line == 0 {
            // No source line — we've left the JIT function
            self.exited = true;
            return Err(DebugError::ProcessExited(0));
        }

        Ok(line)
    }

    fn read_register(&self, preg: u8) -> Result<u64, DebugError> {
        let thread = self.process.selected_thread();
        let frame = thread.selected_frame();
        let reg_name = kajit::harness::AllocationMap::reg_name(preg);
        let value = frame
            .find_register(reg_name)
            .ok_or_else(|| DebugError::LldbError(format!("register {reg_name} not found")))?;
        Ok(value.value_as_unsigned(0))
    }

    fn read_memory(&self, addr: u64, size: usize) -> Result<Vec<u8>, DebugError> {
        let mut buf = vec![0u8; size];
        self.process
            .read_memory(addr, &mut buf)
            .map_err(|e| DebugError::LldbError(format!("read_memory: {e}")))?;
        Ok(buf)
    }

    fn read_sp(&self) -> Result<u64, DebugError> {
        let frame = self.process.selected_thread().selected_frame();
        Ok(frame.sp())
    }

    fn read_pc(&self) -> Result<u64, DebugError> {
        let frame = self.process.selected_thread().selected_frame();
        Ok(frame.pc())
    }

    fn has_exited(&self) -> bool {
        self.exited || !self.process.is_alive()
    }

    fn current_source_line(&self) -> Result<u32, DebugError> {
        let frame = self.process.selected_thread().selected_frame();
        Ok(frame.line_entry().map(|le| le.line()).unwrap_or(0))
    }

    fn disassemble_around_pc(&self, context: usize) -> Result<String, DebugError> {
        let thread = self.process.selected_thread();
        let frame = thread.selected_frame();
        let pc = frame.pc();

        // Use LLDB's command interpreter to get disassembly
        // (the SB API's frame.disassemble() returns the entire function)
        let _ci = self.debugger.command_interpreter();
        let _cmd = format!(
            "disassemble -s 0x{:x} -c {}",
            pc.saturating_sub((context * 4) as u64),
            context * 2 + 1
        );

        // For now, use frame.disassemble() and extract the relevant lines
        let full_disasm = frame.disassemble();
        let mut lines: Vec<&str> = Vec::new();
        let pc_str = format!("0x{:x}", pc);
        let mut found_idx = None;

        for (i, line) in full_disasm.lines().enumerate() {
            if line.contains(&pc_str) || line.contains("->") {
                found_idx = Some(i);
            }
            lines.push(line);
        }

        if let Some(idx) = found_idx {
            let start = idx.saturating_sub(context);
            let end = (idx + context + 1).min(lines.len());
            Ok(lines[start..end]
                .iter()
                .map(|l| {
                    if l.contains(&pc_str) {
                        format!("→ {l}")
                    } else {
                        format!("  {l}")
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"))
        } else {
            // Fallback: just show a few lines around the middle
            let mid = lines.len() / 2;
            let start = mid.saturating_sub(context);
            let end = (mid + context + 1).min(lines.len());
            Ok(lines[start..end]
                .iter()
                .map(|l| format!("  {l}"))
                .collect::<Vec<_>>()
                .join("\n"))
        }
    }
}

impl Drop for LldbJitDebugger {
    fn drop(&mut self) {
        if self.process.is_alive() {
            let _ = self.process.kill();
        }
        SBDebugger::terminate();
    }
}
