//! LLDB implementation of the JitDebugger trait for lockstep debugging.

use kajit::lockstep::{DebugError, JitDebugger};
use lldb::*;

/// LLDB-based JIT debugger that drives a standalone harness process.
pub struct LldbJitDebugger {
    debugger: SBDebugger,
    target: SBTarget,
    process: SBProcess,
    exited: bool,
    /// Address range of kajit_decode function (to detect when we leave it)
    func_start: u64,
    func_end: u64,
}

impl LldbJitDebugger {
    /// Launch a harness executable under LLDB, stopping at the `kajit_decode` entry.
    pub fn launch(harness_path: &str, input_hex: &str) -> Result<Self, DebugError> {
        SBDebugger::initialize();

        let debugger = SBDebugger::create(false);
        debugger.set_asynchronous(false); // synchronous mode — step calls block

        let target = debugger.create_target_simple(harness_path).ok_or_else(|| {
            DebugError::LldbError(format!("failed to create target for {harness_path}"))
        })?;

        // Set breakpoint at kajit_decode
        let bp = target.breakpoint_create_by_name("kajit_decode");
        if !bp.is_valid() {
            return Err(DebugError::LldbError(
                "failed to set breakpoint on kajit_decode".into(),
            ));
        }

        // Launch with the hex input as argument
        let launch_info = SBLaunchInfo::new();
        launch_info.set_arguments([input_hex].into_iter(), false);

        let process = target
            .launch(launch_info)
            .map_err(|e| DebugError::LldbError(format!("launch failed: {e}")))?;

        // Process should stop at our breakpoint
        if !process.is_stopped() {
            return Err(DebugError::LldbError(
                "process did not stop at breakpoint".into(),
            ));
        }

        let thread = process.selected_thread();
        let frame = thread.selected_frame();
        let stop_reason = thread.stop_reason();
        let func_start = frame.pc();

        // Get the function size from the disassembly or use a reasonable default
        // We'll detect function exit by checking if PC leaves the kajit_decode symbol
        let func_end = func_start + 0x10000; // generous upper bound

        eprintln!(
            "[lldb] launched, stopped at kajit_decode @ 0x{:x} (reason: {:?})",
            func_start, stop_reason
        );

        Ok(Self {
            debugger,
            target,
            process,
            exited: false,
            func_start,
            func_end,
        })
    }
}

impl JitDebugger for LldbJitDebugger {
    fn step_to_next_source_line(&mut self) -> Result<u32, DebugError> {
        if self.exited {
            return Err(DebugError::ProcessExited(0));
        }

        let thread = self.process.selected_thread();

        // Step one instruction
        thread
            .step_instruction(false) // step_over=false — step INTO calls
            .map_err(|e| DebugError::LldbError(format!("step_instruction: {e}")))?;

        if !self.process.is_stopped() {
            self.exited = true;
            return Err(DebugError::ProcessExited(self.process.exit_status()));
        }

        let frame = thread.selected_frame();
        let pc = frame.pc();

        // Check if we've left the function (return instruction)
        if pc < self.func_start || pc >= self.func_end {
            self.exited = true;
            return Err(DebugError::ProcessExited(0));
        }

        // Return 0 for line (no DWARF line tracking yet — we're stepping by instruction)
        Ok(0)
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
        let thread = self.process.selected_thread();
        let frame = thread.selected_frame();
        Ok(frame.sp())
    }

    fn read_pc(&self) -> Result<u64, DebugError> {
        let thread = self.process.selected_thread();
        let frame = thread.selected_frame();
        Ok(frame.pc())
    }

    fn has_exited(&self) -> bool {
        self.exited || !self.process.is_alive()
    }

    fn current_source_line(&self) -> Result<u32, DebugError> {
        let thread = self.process.selected_thread();
        let frame = thread.selected_frame();
        Ok(frame.line_entry().map(|le| le.line()).unwrap_or(0))
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
