//! LLDB implementation of the JitDebugger trait for lockstep debugging.

use kajit::lockstep::{DebugError, JitDebugger};
use lldb::*;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

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
        configure_lldb_debugserver_path();
        let init = SBDebugger::initialize_with_error_handling();
        if !init.is_success() {
            return Err(DebugError::LldbError(format!(
                "failed to initialize LLDB: {init}"
            )));
        }

        let debugger = SBDebugger::create(false);
        debugger.set_asynchronous(false); // synchronous mode

        let target = debugger
            .create_target(harness_path, None, None, true)
            .map_err(|e| {
                DebugError::LldbError(format!("failed to create target for {harness_path}: {e}"))
            })?;
        debugger.set_selected_target(&target);

        // Break on the decoder entrypoint itself. Source breakpoints can look
        // valid before launch while still resolving to zero locations.
        let bp = target.breakpoint_create_by_name("kajit_decode");
        if !bp.is_valid() || bp.locations().next().is_none() {
            return Err(DebugError::LldbError(format!(
                "failed to resolve breakpoint on kajit_decode in {harness_path}"
            )));
        }

        let launch_info = SBLaunchInfo::new();
        let exe_file = target.executable().ok_or_else(|| {
            DebugError::LldbError(format!(
                "target executable missing after create_target for {harness_path}"
            ))
        })?;
        launch_info.set_arguments([input_hex], false);
        launch_info.set_executable_file(&exe_file, true);

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

    pub fn execute_command(&self, command: &str) -> Result<String, DebugError> {
        self.debugger
            .execute_command(command)
            .map(|out| out.to_owned())
            .map_err(DebugError::LldbError)
    }

    pub fn read_register_by_name(&self, name: &str) -> Result<u64, DebugError> {
        let frame = self.process.selected_thread().selected_frame();
        let value = frame
            .find_register(name)
            .ok_or_else(|| DebugError::LldbError(format!("register {name} not found")))?;
        Ok(value.value_as_unsigned(0))
    }

    pub fn backtrace(&self) -> Result<String, DebugError> {
        self.execute_command("bt")
    }

    pub fn source_info(&self) -> Result<String, DebugError> {
        let pc = self.read_pc()?;
        let frame_info = self.execute_command("frame info")?;
        let source_info = self.execute_command(&format!("source info --address 0x{pc:x}"))?;
        Ok(format!("{frame_info}\n{source_info}"))
    }
}

fn configure_lldb_debugserver_path() {
    if env::var_os("LLDB_DEBUGSERVER_PATH").is_some() {
        return;
    }

    if let Some(path) = find_lldb_server_path() {
        // This runs before LLDB is initialized, so there are no concurrent
        // environment readers in-process yet.
        unsafe {
            env::set_var("LLDB_DEBUGSERVER_PATH", path);
        }
    }
}

fn find_lldb_server_path() -> Option<PathBuf> {
    find_lldb_server_on_path()
        .or_else(|| find_lldb_server_in_dir(Path::new("/usr/bin")))
        .or_else(|| find_lldb_server_under_prefix(Path::new("/usr/lib")))
        .or_else(|| find_lldb_server_under_prefix(Path::new("/lib")))
}

fn find_lldb_server_on_path() -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path).find_map(|dir| find_lldb_server_in_dir(&dir))
}

fn find_lldb_server_under_prefix(prefix: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(prefix).ok()?;
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("llvm-"))
        })
        .find_map(|llvm_dir| find_lldb_server_in_dir(&llvm_dir.join("bin")))
}

fn find_lldb_server_in_dir(dir: &Path) -> Option<PathBuf> {
    let direct = dir.join("lldb-server");
    if direct.is_file() {
        return Some(direct);
    }

    let entries = fs::read_dir(dir).ok()?;
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("lldb-server"))
        })
}

impl JitDebugger for LldbJitDebugger {
    fn step_instruction_over(&mut self) -> Result<(), DebugError> {
        if self.exited {
            return Err(DebugError::ProcessExited(0));
        }

        // Spawn a watchdog thread that interrupts the process if step_instruction
        // (step over) blocks too long — e.g. when the called function has an
        // infinite loop.
        let done = Arc::new(AtomicBool::new(false));
        let done2 = done.clone();
        let process_clone = self.process.clone();
        let watchdog = std::thread::spawn(move || {
            // Check every 100ms so we don't overshoot by much
            for _ in 0..50 {
                std::thread::sleep(Duration::from_millis(100));
                if done2.load(Ordering::Relaxed) {
                    return false; // step completed in time
                }
            }
            // 5 seconds elapsed — interrupt the process
            let _ = process_clone.stop();
            true
        });

        let thread = self.process.selected_thread();
        let step_result = thread.step_instruction(true);
        done.store(true, Ordering::Relaxed);
        let timed_out = watchdog.join().unwrap_or(false);

        step_result.map_err(|e| DebugError::LldbError(format!("step_instruction: {e}")))?;

        if timed_out {
            // The watchdog interrupted the process. Report where we are.
            let pc = self.process.selected_thread().selected_frame().pc();
            let disasm = self.disassemble_around_pc(8).unwrap_or_default();
            return Err(DebugError::Timeout(format!(
                "step_instruction_over blocked for >5s (likely infinite loop in called function). \
                 pc=0x{pc:x}\n\nDisassembly:\n{disasm}"
            )));
        }

        if !self.process.is_stopped() {
            self.exited = true;
            return Err(DebugError::ProcessExited(self.process.exit_status()));
        }

        let thread = self.process.selected_thread();
        let stop_reason = thread.stop_reason();
        if stop_reason == lldb::StopReason::Signal || stop_reason == lldb::StopReason::Exception {
            self.exited = true;
            let status = self.process.exit_status();
            let sig = if self.process.is_alive() {
                match stop_reason {
                    lldb::StopReason::Signal => status,
                    lldb::StopReason::Exception => 11,
                    _ => status,
                }
            } else {
                status
            };
            return Err(DebugError::ProcessSignaled(sig));
        }

        Ok(())
    }

    fn step_to_next_source_line(&mut self) -> Result<u32, DebugError> {
        if self.exited {
            return Err(DebugError::ProcessExited(0));
        }

        let start_line = self.current_source_line().unwrap_or(0);
        for _ in 0..4096 {
            self.step_instruction_over()?;
            let line = self.current_source_line().unwrap_or(0);
            if line == 0 {
                self.exited = true;
                return Err(DebugError::ProcessExited(0));
            }
            if line != start_line {
                return Ok(line);
            }
        }

        Err(DebugError::LldbError(format!(
            "instruction stepping did not leave source line {start_line}"
        )))
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
