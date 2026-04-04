use std::collections::HashMap;
#[cfg(feature = "lldb")]
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
#[cfg(feature = "lldb")]
use kajit::harness::{LocationMap, LocationTracker, VRegLocation};
#[cfg(feature = "lldb")]
use kajit::lockstep::{DebugError as JitDebugError, Divergence, JitDebugger, LockstepResult};
#[cfg(feature = "lldb")]
use kajit_lir::{BinOpKind, LinearOp};
use kajit_mir::cfg_mir::BlockId;
use kajit_mir::{DebuggerSession, DebuggerState, RunUntilTarget, StepEvent};
use rust_mcp_sdk::macros::{JsonSchema, mcp_tool};
use rust_mcp_sdk::mcp_server::{McpServerOptions, ServerHandler, server_runtime};
use rust_mcp_sdk::schema::{
    CallToolError, CallToolRequestParams, CallToolResult, Implementation, InitializeResult,
    LATEST_PROTOCOL_VERSION, ListToolsResult, PaginatedRequestParams, RpcError, ServerCapabilities,
    ServerCapabilitiesTools,
};
use rust_mcp_sdk::{McpServer, StdioTransport, ToMcpServerHandler, TransportOptions, tool_box};
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};

#[cfg(feature = "lldb")]
use crate::lldb_debugger::LldbJitDebugger;

#[mcp_tool(
    name = "session_new",
    description = "Create a new CFG-MIR debugger session from a file."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SessionNewTool {
    /// Path to a CFG-MIR text file
    cfg_mir_path: String,
    /// Input bytes as hex string (e.g. '8101' or '[0x81, 0x01]')
    #[serde(default)]
    input_hex: Option<String>,
}

#[mcp_tool(
    name = "session_close",
    description = "Close and remove a debugger session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SessionCloseTool {
    /// Session identifier returned by session_new.
    session_id: u64,
}

#[mcp_tool(
    name = "session_step",
    description = "Step a session forward by one or more operations."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SessionStepTool {
    /// Session identifier returned by session_new.
    session_id: u64,
    /// Number of forward steps.
    #[serde(default)]
    count: Option<u64>,
}

#[mcp_tool(
    name = "session_back",
    description = "Step a session backward by one or more recorded steps."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SessionBackTool {
    /// Session identifier returned by session_new.
    session_id: u64,
    /// Number of backwards steps.
    #[serde(default)]
    count: Option<u64>,
}

#[mcp_tool(
    name = "session_run_until",
    description = "Run forward until block/trap/return or max step budget."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SessionRunUntilTool {
    /// Session identifier returned by session_new.
    session_id: u64,
    /// Target block ID. Mutually exclusive with trap/return.
    #[serde(default)]
    block_id: Option<u64>,
    /// Stop when a trap occurs. Mutually exclusive with block_id/return.
    #[serde(default)]
    trap: Option<bool>,
    /// Stop when function returns. Mutually exclusive with block_id/trap.
    #[serde(default)]
    until_return: Option<bool>,
    /// Maximum number of steps.
    #[serde(default)]
    max_steps: Option<u64>,
}

#[mcp_tool(
    name = "session_state",
    description = "Get deterministic full state snapshot for one session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SessionStateTool {
    /// Session identifier returned by session_new.
    session_id: u64,
}

#[mcp_tool(
    name = "session_inspect_vreg",
    description = "Read one virtual register by index."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SessionInspectVregTool {
    /// Session identifier returned by session_new.
    session_id: u64,
    /// Virtual register index.
    vreg: u64,
}

#[mcp_tool(
    name = "session_inspect_output",
    description = "Read a byte range from output memory."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SessionInspectOutputTool {
    /// Session identifier returned by session_new.
    session_id: u64,
    /// Start offset.
    #[serde(default)]
    start: Option<u64>,
    /// Number of bytes to read.
    #[serde(default)]
    len: Option<u64>,
}

#[mcp_tool(
    name = "debug_session_new",
    description = "Compile a decoder, generate a standalone harness, launch LLDB, and create a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionNewTool {
    /// Format: postcard
    format: String,
    /// Type to compile (e.g. u32, ScalarVec, BorrowedHeader)
    ty: String,
    /// Input bytes as hex string (e.g. '8101' or '[0x81, 0x01]')
    input_hex: String,
}

#[mcp_tool(
    name = "debug_session_close",
    description = "Close and remove a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionCloseTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
}

#[mcp_tool(
    name = "debug_session_step",
    description = "Step a persistent lockstep debug session forward by one or more CFG-MIR operations."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionStepTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
    /// Number of lockstep op steps.
    #[serde(default)]
    count: Option<u64>,
}

#[mcp_tool(
    name = "debug_session_state",
    description = "Get combined LLDB + interpreter state for a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionStateTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
}

#[mcp_tool(
    name = "debug_session_disassemble",
    description = "Disassemble around the current PC for a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionDisassembleTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
    /// Number of instructions of context before/after the PC.
    #[serde(default)]
    context: Option<u64>,
}

#[mcp_tool(
    name = "debug_session_registers",
    description = "Read named machine registers for a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionRegistersTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
    /// Optional comma-separated register names (defaults to key decoder registers).
    #[serde(default)]
    names: Option<String>,
}

#[mcp_tool(
    name = "debug_session_memory",
    description = "Read a byte range from the debuggee process memory for a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionMemoryTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
    /// Start address to read from.
    address: u64,
    /// Number of bytes to read.
    #[serde(default)]
    len: Option<u64>,
}

#[mcp_tool(
    name = "debug_session_backtrace",
    description = "Get a backtrace for the currently stopped frame in a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionBacktraceTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
}

#[mcp_tool(
    name = "debug_session_source_info",
    description = "Get source and frame info for the current PC in a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionSourceInfoTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
}

#[mcp_tool(
    name = "debug_session_lldb",
    description = "Run a raw LLDB command against a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionLldbTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
    /// Raw LLDB command.
    command: String,
}

tool_box!(
    MirTools,
    [
        SessionNewTool,
        SessionCloseTool,
        SessionStepTool,
        SessionBackTool,
        SessionRunUntilTool,
        SessionStateTool,
        SessionInspectVregTool,
        SessionInspectOutputTool,
        DebugSessionNewTool,
        DebugSessionCloseTool,
        DebugSessionStepTool,
        DebugSessionStateTool,
        DebugSessionDisassembleTool,
        DebugSessionRegistersTool,
        DebugSessionMemoryTool,
        DebugSessionBacktraceTool,
        DebugSessionSourceInfoTool,
        DebugSessionLldbTool
    ]
);

#[derive(Default)]
struct ServerState {
    sessions: HashMap<u64, DebuggerSession>,
    next_session_id: u64,
    #[cfg(feature = "lldb")]
    debug_sessions: HashMap<u64, DebugDiffSession>,
    #[cfg(feature = "lldb")]
    next_debug_session_id: u64,
}

#[derive(Clone, Default)]
struct MirHandler {
    state: Arc<Mutex<ServerState>>,
}

impl MirHandler {
    fn lock_state(&self) -> Result<MutexGuard<'_, ServerState>, String> {
        self.state
            .lock()
            .map_err(|_| "internal error: debugger state mutex poisoned".to_owned())
    }

    fn call_tool(
        &self,
        name: &str,
        args: &JsonMap<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        match name {
            "session_new" => self.session_new(args),
            "session_close" => self.session_close(args),
            "session_step" => self.session_step(args),
            "session_back" => self.session_back(args),
            "session_run_until" => self.session_run_until(args),
            "session_state" => self.session_state(args),
            "session_inspect_vreg" => self.session_inspect_vreg(args),
            "session_inspect_output" => self.session_inspect_output(args),
            "debug_session_new" => self.debug_session_new(args),
            "debug_session_close" => self.debug_session_close(args),
            "debug_session_step" => self.debug_session_step(args),
            "debug_session_state" => self.debug_session_state(args),
            "debug_session_disassemble" => self.debug_session_disassemble(args),
            "debug_session_registers" => self.debug_session_registers(args),
            "debug_session_memory" => self.debug_session_memory(args),
            "debug_session_backtrace" => self.debug_session_backtrace(args),
            "debug_session_source_info" => self.debug_session_source_info(args),
            "debug_session_lldb" => self.debug_session_lldb(args),
            other => Err(format!("unknown tool: {other}")),
        }
    }

    fn session_new(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        let mir_path = arg_str(args, "cfg_mir_path")?;
        let mir_text = std::fs::read_to_string(&mir_path)
            .map_err(|e| format!("failed to read {mir_path}: {e}"))?;
        let input_hex = arg_opt_str(args, "input_hex").unwrap_or_default();
        let input = parse_hex_input(&input_hex)?;
        let program = kajit_mir_text::parse_cfg_mir(&mir_text).map_err(|e| e.to_string())?;
        let session = DebuggerSession::new(&program, &input).map_err(|e| e.to_string())?;

        let mut state = self.lock_state()?;
        if state.next_session_id == 0 {
            state.next_session_id = 1;
        }
        let session_id = state.next_session_id;
        state.next_session_id += 1;
        state.sessions.insert(session_id, session);

        let snapshot = state
            .sessions
            .get(&session_id)
            .expect("inserted session should exist")
            .state();
        let mut md = format!(
            "Session **{}** created (input: `{}`)\n\n",
            session_id,
            encode_hex(&input)
        );
        md.push_str(&format_state_markdown(&snapshot));
        Ok(json!({ "text": md }))
    }

    fn session_close(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        let session_id = arg_u64(args, "session_id")?;
        let mut state = self.lock_state()?;
        let removed = state.sessions.remove(&session_id).is_some();
        Ok(json!({
            "session_id": session_id,
            "closed": removed,
        }))
    }

    fn session_step(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        let session_id = arg_u64(args, "session_id")?;
        let count = arg_opt_u64(args, "count").unwrap_or(1) as usize;

        let mut state = self.lock_state()?;
        let session = state
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| format!("unknown session_id: {session_id}"))?;

        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            let event = session.step_forward().map_err(|e| e.to_string())?;
            events.push(event_json(&event));
        }

        Ok(json!({
            "session_id": session_id,
            "events": events,
            "state": state_json(&session.state()),
        }))
    }

    fn session_back(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        let session_id = arg_u64(args, "session_id")?;
        let count = arg_opt_u64(args, "count").unwrap_or(1) as usize;

        let mut state = self.lock_state()?;
        let session = state
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| format!("unknown session_id: {session_id}"))?;

        let mut undone = 0usize;
        for _ in 0..count {
            if session.step_back() {
                undone += 1;
            } else {
                break;
            }
        }

        Ok(json!({
            "session_id": session_id,
            "undone": undone,
            "state": state_json(&session.state()),
        }))
    }

    fn session_run_until(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        let session_id = arg_u64(args, "session_id")?;
        let block_id = arg_opt_u64(args, "block_id");
        let want_trap = arg_opt_bool(args, "trap").unwrap_or(false);
        let want_return = arg_opt_bool(args, "until_return").unwrap_or(false);
        let max_steps = arg_opt_u64(args, "max_steps").unwrap_or(10_000) as usize;

        let target = match (block_id, want_trap, want_return) {
            (Some(block), false, false) => RunUntilTarget::Block(BlockId(block as u32)),
            (None, true, false) => RunUntilTarget::Trap,
            (None, false, true) => RunUntilTarget::Return,
            (None, false, false) => {
                return Err("one of block_id/trap/return must be specified".to_owned());
            }
            _ => {
                return Err(
                    "block_id/trap/return are mutually exclusive (pick exactly one)".to_owned(),
                );
            }
        };

        let mut state = self.lock_state()?;
        let session = state
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| format!("unknown session_id: {session_id}"))?;

        let events = session
            .run_until(target, max_steps)
            .map_err(|e| e.to_string())?;

        let mut md = format!("**Session {}** — {} steps:\n\n", session_id, events.len());
        for event in &events {
            md.push_str(&format_event_markdown(event));
            md.push('\n');
        }
        md.push_str(&format!("\n**Final state:**\n"));
        md.push_str(&format_state_markdown(&session.state()));
        Ok(json!({ "text": md }))
    }

    fn session_state(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        let session_id = arg_u64(args, "session_id")?;
        let state = self
            .lock_state()?
            .sessions
            .get(&session_id)
            .ok_or_else(|| format!("unknown session_id: {session_id}"))?
            .state();
        let mut md = format!("**Session {}**\n\n", session_id);
        md.push_str(&format_state_markdown(&state));
        Ok(json!({ "text": md }))
    }

    fn session_inspect_vreg(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        let session_id = arg_u64(args, "session_id")?;
        let vreg = arg_u64(args, "vreg")? as usize;
        let state = self.lock_state()?;
        let session = state
            .sessions
            .get(&session_id)
            .ok_or_else(|| format!("unknown session_id: {session_id}"))?;
        let value = session.inspect_vreg(vreg);
        Ok(json!({ "text": format!("v{} = {} (0x{:x})", vreg, value, value) }))
    }

    fn session_inspect_output(
        &self,
        args: &JsonMap<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        let session_id = arg_u64(args, "session_id")?;
        let start = arg_opt_u64(args, "start").unwrap_or(0) as usize;
        let len = arg_opt_u64(args, "len").unwrap_or(64) as usize;
        let state = self.lock_state()?;
        let session = state
            .sessions
            .get(&session_id)
            .ok_or_else(|| format!("unknown session_id: {session_id}"))?;
        let bytes = session.inspect_output(start, len);
        Ok(
            json!({ "text": format!("output[{}..{}]: `{}`", start, start + bytes.len(), encode_hex(&bytes)) }),
        )
    }

    fn debug_session_new(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let format = arg_str(args, "format")?;
            let ty = arg_str(args, "ty")?;
            let input_hex = arg_str(args, "input_hex")?;
            let session = DebugDiffSession::new(&format, &ty, &input_hex)?;

            let mut state = self.lock_state()?;
            if state.next_debug_session_id == 0 {
                state.next_debug_session_id = 1;
            }
            let session_id = state.next_debug_session_id;
            state.next_debug_session_id += 1;
            state.debug_sessions.insert(session_id, session);

            let session = state
                .debug_sessions
                .get(&session_id)
                .expect("inserted debug session should exist");
            Ok(json!({ "text": session.snapshot_markdown(session_id) }))
        }
    }

    fn debug_session_close(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let mut state = self.lock_state()?;
            let removed = state.debug_sessions.remove(&session_id).is_some();
            Ok(json!({
                "session_id": session_id,
                "closed": removed,
            }))
        }
    }

    fn debug_session_step(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let count = arg_opt_u64(args, "count").unwrap_or(1) as usize;

            let mut state = self.lock_state()?;
            let session = state
                .debug_sessions
                .get_mut(&session_id)
                .ok_or_else(|| format!("unknown session_id: {session_id}"))?;

            let mut out = format!("**Debug session {}**\n\n", session_id);
            for _ in 0..count {
                let step = session.step_forward()?;
                out.push_str(&step);
                out.push('\n');
                if !session.is_running() {
                    break;
                }
            }
            out.push('\n');
            out.push_str(&session.snapshot_markdown(session_id));
            Ok(json!({ "text": out }))
        }
    }

    fn debug_session_state(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let state = self.lock_state()?;
            let session = state
                .debug_sessions
                .get(&session_id)
                .ok_or_else(|| format!("unknown session_id: {session_id}"))?;
            Ok(json!({ "text": session.snapshot_markdown(session_id) }))
        }
    }

    fn debug_session_disassemble(
        &self,
        args: &JsonMap<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let context = arg_opt_u64(args, "context").unwrap_or(4) as usize;
            let state = self.lock_state()?;
            let session = state
                .debug_sessions
                .get(&session_id)
                .ok_or_else(|| format!("unknown session_id: {session_id}"))?;
            let text = session
                .debugger
                .disassemble_around_pc(context)
                .map_err(|e| e.to_string())?;
            Ok(json!({ "text": text }))
        }
    }

    fn debug_session_registers(
        &self,
        args: &JsonMap<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let names = parse_debug_register_names(arg_opt_str(args, "names").as_deref());
            let state = self.lock_state()?;
            let session = state
                .debug_sessions
                .get(&session_id)
                .ok_or_else(|| format!("unknown session_id: {session_id}"))?;

            let mut values = serde_json::Map::new();
            let mut text = String::new();
            for name in names {
                let value = session
                    .debugger
                    .read_register_by_name(&name)
                    .map_err(|e| e.to_string())?;
                values.insert(name.clone(), json!(value));
                text.push_str(&format!("{name}=0x{value:x} ({value})\n"));
            }

            Ok(json!({
                "text": text,
                "registers": values,
            }))
        }
    }

    fn debug_session_memory(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let address = arg_u64(args, "address")?;
            let len = arg_opt_u64(args, "len").unwrap_or(64) as usize;
            let state = self.lock_state()?;
            let session = state
                .debug_sessions
                .get(&session_id)
                .ok_or_else(|| format!("unknown session_id: {session_id}"))?;
            let bytes = session
                .debugger
                .read_memory(address, len)
                .map_err(|e| e.to_string())?;
            Ok(json!({
                "text": format!("mem[0x{address:x}..0x{:x}] = `{}`", address + bytes.len() as u64, encode_hex(&bytes)),
                "address": address,
                "len": bytes.len(),
                "hex": encode_hex(&bytes),
            }))
        }
    }

    fn debug_session_backtrace(
        &self,
        args: &JsonMap<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let state = self.lock_state()?;
            let session = state
                .debug_sessions
                .get(&session_id)
                .ok_or_else(|| format!("unknown session_id: {session_id}"))?;
            let text = session.debugger.backtrace().map_err(|e| e.to_string())?;
            Ok(json!({ "text": text }))
        }
    }

    fn debug_session_source_info(
        &self,
        args: &JsonMap<String, JsonValue>,
    ) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let state = self.lock_state()?;
            let session = state
                .debug_sessions
                .get(&session_id)
                .ok_or_else(|| format!("unknown session_id: {session_id}"))?;
            let pc = session.debugger.read_pc().map_err(|e| e.to_string())?;
            let dwarf_line = session
                .debugger
                .current_source_line()
                .map_err(|e| e.to_string())?;
            let cfg_line = session.current_line_text(dwarf_line);
            let lldb = session.debugger.source_info().map_err(|e| e.to_string())?;
            Ok(json!({
                "text": format!("pc=0x{pc:x}\ndwarf_line={dwarf_line}\ncfg=`{cfg_line}`\n\n{lldb}"),
                "pc": pc,
                "dwarf_line": dwarf_line,
                "cfg_line": cfg_line,
            }))
        }
    }

    fn debug_session_lldb(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let command = arg_str(args, "command")?;
            let state = self.lock_state()?;
            let session = state
                .debug_sessions
                .get(&session_id)
                .ok_or_else(|| format!("unknown session_id: {session_id}"))?;
            let text = session
                .debugger
                .execute_command(&command)
                .map_err(|e| e.to_string())?;
            Ok(json!({ "text": text }))
        }
    }
}

#[async_trait]
impl ServerHandler for MirHandler {
    async fn handle_list_tools_request(
        &self,
        _params: Option<PaginatedRequestParams>,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<ListToolsResult, RpcError> {
        Ok(ListToolsResult {
            tools: MirTools::tools(),
            meta: None,
            next_cursor: None,
        })
    }

    async fn handle_call_tool_request(
        &self,
        params: CallToolRequestParams,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<CallToolResult, CallToolError> {
        let args = params.arguments.unwrap_or_default();
        let result = match self.call_tool(params.name.as_str(), &args) {
            Ok(payload) => call_tool_ok(payload),
            Err(message) => call_tool_err(&message),
        };
        Ok(result)
    }
}

#[cfg(feature = "lldb")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DebugDiffSessionStatus {
    Running,
    Diverged,
    Completed,
}

#[cfg(feature = "lldb")]
struct DebugDiffSession {
    format: String,
    ty: String,
    input: Vec<u8>,
    exe_path: PathBuf,
    listing_path: PathBuf,
    cfg_program: kajit_mir::cfg_mir::Program,
    location_map: LocationMap,
    location_tracker: LocationTracker,
    listing_lines: Vec<String>,
    interpreter: DebuggerSession,
    debugger: LldbJitDebugger,
    op_to_line: HashMap<(u32, bool, usize), u32>,
    prev_dwarf_line: u32,
    jit_steps: usize,
    verified: HashMap<u32, (u64, u32, usize)>,
    status: DebugDiffSessionStatus,
    divergence: Option<Divergence>,
}

#[cfg(feature = "lldb")]
impl DebugDiffSession {
    fn new(format: &str, ty: &str, input_hex: &str) -> Result<Self, String> {
        let kind = match format {
            "postcard" => kajit::DecoderKind::Postcard,
            other => return Err(format!("unknown format '{other}', expected 'postcard'")),
        };

        let shape = super::resolve_shape(ty);
        let pipeline_opts = kajit::PipelineOptions::from_env();
        let artifacts = kajit::compile_pipeline(shape, kind, &pipeline_opts);

        let output_size = shape.layout.sized_layout().map(|l| l.size()).unwrap_or(0);
        let output_dir = PathBuf::from("/tmp/kajit-harness");
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
            .map_err(|e| format!("error generating harness: {e}"))?;

        let input = parse_hex_input(input_hex)?;
        let debugger = LldbJitDebugger::launch(exe_path.to_str().unwrap(), input_hex)
            .map_err(|e| format!("error launching LLDB: {e}"))?;

        let jit_output_ptr = debugger.read_register(21).map_err(|e| e.to_string())?;
        let jit_root_cursor_arg = if artifacts.decoder.uses_root_cursor_arg() {
            Some(debugger.read_register(2).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let (jit_cursor, jit_input_end) = if let Some(root_cursor_arg) = jit_root_cursor_arg {
            let input_base =
                read_debug_u64(&debugger, root_cursor_arg).map_err(|e| e.to_string())?;
            let input_len =
                read_debug_u64(&debugger, root_cursor_arg + 8).map_err(|e| e.to_string())?;
            (input_base, input_base.wrapping_add(input_len))
        } else {
            (
                debugger.read_register(19).map_err(|e| e.to_string())?,
                debugger.read_register(20).map_err(|e| e.to_string())?,
            )
        };

        let listing_lines = artifacts.decoder.cfg_mir_lines().to_vec();

        let mut interpreter = DebuggerSession::new(&artifacts.cfg_program, &input)
            .map_err(|e| format!("interpreter init: {e}"))?;
        interpreter.input_base_addr = Some(jit_cursor);
        interpreter.input_end_addr = Some(jit_input_end);
        interpreter.output_base_addr = Some(jit_output_ptr);
        if let Some(root_cursor_arg) = jit_root_cursor_arg {
            interpreter.set_root_cursor_arg_addr(root_cursor_arg);
        }

        let mut op_to_line = HashMap::new();
        let mut next_line = 1u32;
        let func = &artifacts.cfg_program.funcs[0];
        for block in &func.blocks {
            for (inst_idx, _) in block.insts.iter().enumerate() {
                op_to_line.insert((block.id.0, false, inst_idx), next_line);
                next_line += 1;
            }
            op_to_line.insert((block.id.0, true, 0), next_line);
            next_line += 1;
        }

        let prev_dwarf_line = debugger.current_source_line().unwrap_or(1);
        let location_tracker =
            LocationTracker::new(&artifacts.location_map, &artifacts.cfg_program);

        Ok(Self {
            format: format.to_owned(),
            ty: ty.to_owned(),
            input,
            exe_path,
            listing_path,
            cfg_program: artifacts.cfg_program,
            location_map: artifacts.location_map,
            location_tracker,
            listing_lines,
            interpreter,
            debugger,
            op_to_line,
            prev_dwarf_line,
            jit_steps: 0,
            verified: HashMap::new(),
            status: DebugDiffSessionStatus::Running,
            divergence: None,
        })
    }

    fn is_running(&self) -> bool {
        self.status == DebugDiffSessionStatus::Running
    }

    fn current_line_text(&self, line: u32) -> String {
        if line > 0 && (line as usize) <= self.listing_lines.len() {
            self.listing_lines[line as usize - 1].clone()
        } else {
            format!("<line {line}>")
        }
    }

    fn finish_with_result(&mut self, result: LockstepResult) -> String {
        self.divergence = result.divergence.clone();
        self.status = if result.completed {
            DebugDiffSessionStatus::Completed
        } else {
            DebugDiffSessionStatus::Diverged
        };
        kajit::lockstep::format_result(&result)
    }

    fn step_forward(&mut self) -> Result<String, String> {
        if self.status != DebugDiffSessionStatus::Running {
            let result = LockstepResult {
                steps: self.jit_steps,
                divergence: self.divergence.clone(),
                completed: self.status == DebugDiffSessionStatus::Completed,
            };
            return Ok(kajit::lockstep::format_result(&result));
        }

        if self.debugger.has_exited() {
            let result = handle_jit_exit(
                &mut self.interpreter,
                &self.op_to_line,
                &self.listing_lines,
                self.jit_steps,
                self.prev_dwarf_line,
                "already exited",
            )
            .map_err(|e| e.to_string())?;
            return Ok(self.finish_with_result(result));
        }

        let dwarf_line = match self.debugger.step_to_next_source_line() {
            Ok(line) => line,
            Err(JitDebugError::ProcessExited(code)) => {
                let result = handle_jit_exit(
                    &mut self.interpreter,
                    &self.op_to_line,
                    &self.listing_lines,
                    self.jit_steps,
                    self.prev_dwarf_line,
                    &format!("exited with code {code}"),
                )
                .map_err(|e| e.to_string())?;
                return Ok(self.finish_with_result(result));
            }
            Err(JitDebugError::ProcessSignaled(sig)) => {
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
                )
                .map_err(|e| e.to_string())?;
                return Ok(self.finish_with_result(result));
            }
            Err(e) => return Err(e.to_string()),
        };

        self.jit_steps += 1;
        let executed_line = self.prev_dwarf_line;
        self.prev_dwarf_line = dwarf_line;

        let mut last_event = None;
        let mut synced = false;
        for _ in 0..500 {
            let pre_loc = self.interpreter.state().location;
            let pre_line = loc_to_line(&self.op_to_line, &pre_loc);

            if pre_line == executed_line && last_event.is_none() {
                let event = self
                    .interpreter
                    .step_forward()
                    .map_err(|e| format!("interpreter step: {e}"))?;
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
                .map_err(|e| format!("interpreter step: {e}"))?;

            let ev_line = loc_to_line(&self.op_to_line, &event.location_before);
            if ev_line == executed_line && last_event.is_none() {
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

        let state = self.interpreter.state();
        let func = &self.cfg_program.funcs[0];
        let loc = &event.location_before;
        let (def_vreg, use_vregs, _) = op_def_uses_and_kind(func, loc);

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

        self.location_tracker.observe_step(
            &self.location_map,
            func,
            executed_line,
            loc,
            &interp_next_loc,
        );

        if interp_next_line != dwarf_line && !state.returned && dwarf_line != 0 {
            let jit_pc = self.debugger.read_pc().map_err(|e| e.to_string())?;
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
                    self.location_tracker
                        .location_for(&self.location_map, d.index() as u32)
                }),
            ) {
                let sp = self.debugger.read_sp().map_err(|e| e.to_string())?;
                let iv = if dst.index() < state.vregs.len() {
                    state.vregs[dst.index()]
                } else {
                    0
                };
                let jv =
                    read_vreg_from_jit(&self.debugger, location, sp).map_err(|e| e.to_string())?;
                vreg_diffs.push(kajit::lockstep::VRegDiff {
                    vreg_index: dst.index() as u32,
                    interpreter_value: iv,
                    jit_value: jv,
                    jit_location: location.clone(),
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
        {
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
            let location = self
                .location_tracker
                .location_for(&self.location_map, dst_idx);
            if let Some(location) = location {
                let interp_value = if dst.index() < state.vregs.len() {
                    state.vregs[dst.index()]
                } else {
                    0
                };

                let sp = self.debugger.read_sp().map_err(|e| e.to_string())?;
                let jit_value =
                    read_vreg_from_jit(&self.debugger, location, sp).map_err(|e| e.to_string())?;

                if jit_value != Some(interp_value) {
                    let mut vreg_diffs = vec![kajit::lockstep::VRegDiff {
                        vreg_index: dst_idx,
                        interpreter_value: interp_value,
                        jit_value,
                        jit_location: location.clone(),
                        matches: false,
                    }];
                    for use_vreg in &use_vregs {
                        let use_idx = use_vreg.index() as u32;
                        if let Some(use_loc) = self
                            .location_tracker
                            .location_for(&self.location_map, use_idx)
                        {
                            let use_interp = if use_vreg.index() < state.vregs.len() {
                                state.vregs[use_vreg.index()]
                            } else {
                                0
                            };
                            let use_jit = read_vreg_from_jit(&self.debugger, use_loc, sp)
                                .map_err(|e| e.to_string())?;
                            vreg_diffs.push(kajit::lockstep::VRegDiff {
                                vreg_index: use_idx,
                                interpreter_value: use_interp,
                                jit_value: use_jit,
                                jit_location: use_loc.clone(),
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
                        VRegLocation::Register(p) => LocationMap::reg_name(*p),
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
                    diag.push_str(&format!("\n  machine code:\n{}\n", disasm));

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
                    VRegLocation::Register(p) => LocationMap::reg_name(*p),
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

    fn snapshot_markdown(&self, session_id: u64) -> String {
        let status = match self.status {
            DebugDiffSessionStatus::Running => "running",
            DebugDiffSessionStatus::Diverged => "diverged",
            DebugDiffSessionStatus::Completed => "completed",
        };
        let pc = self.debugger.read_pc().ok();
        let dwarf_line = self
            .debugger
            .current_source_line()
            .ok()
            .unwrap_or(self.prev_dwarf_line);
        let mut out = format!(
            "**Debug session {}** `{}` `{}`\n\nstatus: **{}** | steps={} | input=`{}`\nexe: `{}`\nlisting: `{}`\n",
            session_id,
            self.format,
            self.ty,
            status,
            self.jit_steps,
            encode_hex(&self.input),
            self.exe_path.display(),
            self.listing_path.display(),
        );
        if let Some(pc) = pc {
            out.push_str(&format!(
                "pc=0x{pc:x} | dwarf_line={} | source=`{}`\n",
                dwarf_line,
                self.current_line_text(dwarf_line)
            ));
        }
        out.push('\n');
        let state = self.interpreter.state();
        out.push_str(&format_state_markdown(&state));
        if let Some(provenance) = self.current_provenance_markdown(&state) {
            out.push('\n');
            out.push_str("**provenance**\n");
            out.push_str(&provenance);
        }
        if let Some(div) = &self.divergence {
            out.push_str("\n**last divergence**\n");
            out.push_str(&div.source_line);
            out.push('\n');
        }
        out
    }

    fn current_provenance_markdown(&self, state: &DebuggerState) -> Option<String> {
        let loc = &state.location;
        let func = self.cfg_program.funcs.first()?;
        let block = func.blocks.get(loc.block.index())?;
        let op_id = if loc.at_terminator {
            kajit_mir::cfg_mir::OpId::Term(block.term)
        } else {
            kajit_mir::cfg_mir::OpId::Inst(*block.insts.get(loc.next_inst_index)?)
        };
        let cfg_line = loc_to_line(&self.op_to_line, loc);
        let mut out = String::new();
        out.push_str(&format!(
            "cfg_line={} | op=`{}`\n",
            cfg_line,
            self.current_line_text(cfg_line)
        ));

        if let Some(scope_id) = self.cfg_program.op_debug_scope(func.lambda_id, op_id) {
            out.push_str(&format!(
                "op_scope: {}\n",
                self.format_scope_chain(scope_id)
            ));
        }
        if let Some(value_id) = self.cfg_program.op_debug_value(func.lambda_id, op_id) {
            out.push_str(&format!(
                "op_value: {}\n",
                self.format_debug_value(value_id)
            ));
        }

        let (def_vreg, use_vregs, _) = op_def_uses_and_kind(func, loc);
        let mut vreg_lines = Vec::new();
        if let Some(vreg) = def_vreg {
            vreg_lines.push(self.format_vreg_provenance(state, "def", vreg));
        }
        for vreg in use_vregs {
            vreg_lines.push(self.format_vreg_provenance(state, "use", vreg));
        }
        if !vreg_lines.is_empty() {
            out.push_str("vregs:\n");
            for line in vreg_lines {
                out.push_str(&line);
                out.push('\n');
            }
        }

        Some(out)
    }

    fn format_vreg_provenance(
        &self,
        state: &DebuggerState,
        role: &str,
        vreg: kajit_ir::VReg,
    ) -> String {
        let mut line = format!(
            "  {} v{}={}",
            role,
            vreg.index(),
            state.vregs.get(vreg.index()).copied().unwrap_or(0)
        );
        if let Some(scope_id) = self.cfg_program.vreg_debug_scope(vreg) {
            line.push_str(&format!(" | scope {}", self.format_scope_chain(scope_id)));
        }
        if let Some(value_id) = self.cfg_program.vreg_debug_value(vreg) {
            line.push_str(&format!(" | value {}", self.format_debug_value(value_id)));
        }
        line
    }

    fn format_scope_chain(&self, scope_id: kajit_ir::DebugScopeId) -> String {
        let mut chain = Vec::new();
        let mut current = Some(scope_id);
        while let Some(id) = current {
            let scope = &self.cfg_program.debug.scopes[id];
            chain.push(format!(
                "@s{} {}",
                id.index(),
                format_scope_kind(&scope.kind)
            ));
            current = scope.parent;
        }
        chain.join(" <- ")
    }

    fn format_debug_value(&self, value_id: kajit_ir::DebugValueId) -> String {
        let value = &self.cfg_program.debug.values[value_id];
        match &value.kind {
            kajit_ir::DebugValueKind::Field { offset } => {
                format!("{} [field offset={}]", value.name, offset)
            }
            kajit_ir::DebugValueKind::Named => format!("{} [named]", value.name),
        }
    }
}

#[cfg(feature = "lldb")]
fn loc_to_line(map: &HashMap<(u32, bool, usize), u32>, loc: &kajit_mir::ProgramLocation) -> u32 {
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

#[cfg(feature = "lldb")]
fn handle_jit_exit(
    session: &mut DebuggerSession,
    op_to_line: &HashMap<(u32, bool, usize), u32>,
    listing_lines: &[String],
    jit_steps: usize,
    last_dwarf_line: u32,
    exit_reason: &str,
) -> Result<LockstepResult, JitDebugError> {
    let mut interp_steps = 0;
    loop {
        let ev = session
            .step_forward()
            .map_err(|e| JitDebugError::LldbError(format!("interpreter step: {e}")))?;
        interp_steps += 1;
        if ev.returned || interp_steps > 10_000 {
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

    if final_state.returned && interp_steps <= 2 {
        return Ok(LockstepResult {
            steps: jit_steps,
            divergence: None,
            completed: true,
        });
    }

    let source_line = format!(
        "\
JIT EARLY EXIT at step {jit_steps}

  JIT {exit_reason} at line {last_dwarf_line}: {last_op}
  interpreter needed {interp_steps} more steps (returned={returned}, at line {interp_line}: {interp_op})",
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

#[cfg(feature = "lldb")]
fn read_vreg_from_jit(
    debugger: &dyn JitDebugger,
    location: &VRegLocation,
    sp: u64,
) -> Result<Option<u64>, JitDebugError> {
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

#[cfg(feature = "lldb")]
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

#[cfg(feature = "lldb")]
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

#[cfg(feature = "lldb")]
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

#[cfg(feature = "lldb")]
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

#[cfg(feature = "lldb")]
fn is_process_local_pointer_intrinsic(func: kajit_ir::IntrinsicFn) -> bool {
    let f = func.0;
    let alloc_persistent = kajit::intrinsics::kajit_alloc_persistent as *const () as usize;
    let alloc_transient = kajit::intrinsics::kajit_alloc_transient as *const () as usize;
    let vec_alloc = kajit::intrinsics::kajit_vec_alloc as *const () as usize;
    let vec_grow = kajit::intrinsics::kajit_vec_grow as *const () as usize;
    let map_build = kajit::intrinsics::kajit_map_build as *const () as usize;
    let string_alloc =
        kajit::intrinsics::kajit_postcard_validate_and_alloc_string as *const () as usize;
    let string_copy = kajit::intrinsics::kajit_string_validate_alloc_copy as *const () as usize;
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

#[cfg(feature = "lldb")]
fn format_scope_kind(kind: &kajit_ir::DebugScopeKind) -> String {
    match kind {
        kajit_ir::DebugScopeKind::LambdaBody { lambda_id } => {
            format!("lambda_body(@{})", lambda_id.index())
        }
        kajit_ir::DebugScopeKind::GammaBranch { branch_index } => {
            format!("gamma_branch({branch_index})")
        }
        kajit_ir::DebugScopeKind::ThetaBody => "theta_body".to_owned(),
        kajit_ir::DebugScopeKind::Synthetic => "synthetic".to_owned(),
    }
}

fn arg_str(args: &JsonMap<String, JsonValue>, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("missing or invalid string argument `{key}`"))
}

fn arg_opt_str(args: &JsonMap<String, JsonValue>, key: &str) -> Option<String> {
    args.get(key)
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
}

fn arg_u64(args: &JsonMap<String, JsonValue>, key: &str) -> Result<u64, String> {
    args.get(key)
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| format!("missing or invalid integer argument `{key}`"))
}

fn arg_opt_u64(args: &JsonMap<String, JsonValue>, key: &str) -> Option<u64> {
    args.get(key).and_then(JsonValue::as_u64)
}

fn arg_opt_bool(args: &JsonMap<String, JsonValue>, key: &str) -> Option<bool> {
    args.get(key).and_then(JsonValue::as_bool)
}

#[cfg(feature = "lldb")]
fn default_debug_register_names() -> Vec<String> {
    [
        "pc", "sp", "fp", "lr", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

#[cfg(feature = "lldb")]
fn parse_debug_register_names(spec: Option<&str>) -> Vec<String> {
    match spec {
        Some(spec) => {
            let names: Vec<_> = spec
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
                .collect();
            if names.is_empty() {
                default_debug_register_names()
            } else {
                names
            }
        }
        None => default_debug_register_names(),
    }
}

#[cfg(feature = "lldb")]
fn read_debug_u64(debugger: &LldbJitDebugger, addr: u64) -> Result<u64, kajit::lockstep::DebugError> {
    let bytes = debugger.read_memory(addr, 8)?;
    if bytes.len() != 8 {
        return Err(kajit::lockstep::DebugError::LldbError(format!(
            "short read at 0x{addr:x}: expected 8 bytes, got {}",
            bytes.len()
        )));
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes);
    Ok(u64::from_le_bytes(raw))
}

fn trap_json(trap: &kajit_mir::InterpreterTrap) -> JsonValue {
    json!({
        "code": trap.code.to_string(),
        "code_num": trap.code as u32,
        "offset": trap.offset,
    })
}

fn state_json(state: &DebuggerState) -> JsonValue {
    // Markdown-friendly output instead of raw JSON
    json!({
        "text": format_state_markdown(state),
    })
}

fn format_state_markdown(state: &DebuggerState) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "**b{}** inst={} {} | cursor={} | steps={}\n",
        state.location.block.0,
        state.location.next_inst_index,
        if state.location.at_terminator {
            "(at term)"
        } else {
            ""
        },
        state.cursor,
        state.step_count,
    ));
    if let Some(trap) = &state.trap {
        s.push_str(&format!(
            "**TRAP**: {:?} at offset {}\n",
            trap.code, trap.offset
        ));
    }
    if state.returned {
        s.push_str("**RETURNED**\n");
    }
    if state.halted {
        s.push_str("**HALTED**\n");
    }
    s.push_str(&format!("output: `{}`\n", encode_hex(&state.output)));

    // Only show non-zero vregs
    let nonzero: Vec<_> = state
        .vregs
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0)
        .collect();
    if !nonzero.is_empty() {
        s.push_str("vregs: ");
        for (i, (idx, val)) in nonzero.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("v{}={}", idx, val));
        }
        s.push('\n');
    }
    s
}

fn format_event_markdown(event: &StepEvent) -> String {
    let mut s = String::new();
    let arrow = if event.location_before.block != event.location_after.block {
        format!(
            "b{}→b{}",
            event.location_before.block.0, event.location_after.block.0
        )
    } else {
        format!("b{}", event.location_before.block.0)
    };
    s.push_str(&format!(
        "#{} [{}] `{}` cursor={}",
        event.step_index, arrow, event.detail, event.cursor_after,
    ));
    if let Some(trap) = &event.trap {
        s.push_str(&format!(" **TRAP {:?}**", trap.code));
    }
    if event.returned {
        s.push_str(" **RETURN**");
    }
    s
}

fn event_json(event: &StepEvent) -> JsonValue {
    let trap = event
        .trap
        .map(|trap| trap_json(&trap))
        .unwrap_or(JsonValue::Null);
    json!({
        "step_index": event.step_index,
        "kind": format!("{:?}", event.kind),
        "location_before": {
            "block": event.location_before.block.0,
            "next_inst_index": event.location_before.next_inst_index,
            "at_terminator": event.location_before.at_terminator,
        },
        "location_after": {
            "block": event.location_after.block.0,
            "next_inst_index": event.location_after.next_inst_index,
            "at_terminator": event.location_after.at_terminator,
        },
        "cursor_before": event.cursor_before,
        "cursor_after": event.cursor_after,
        "trap": trap,
        "returned": event.returned,
        "halted_after": event.halted_after,
        "detail": event.detail,
    })
}

fn call_tool_ok(payload: JsonValue) -> CallToolResult {
    // If payload has a "text" field, use it directly as markdown
    let text = if let Some(t) = payload.get("text").and_then(|v| v.as_str()) {
        t.to_string()
    } else {
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| payload.to_string())
    };
    CallToolResult::text_content(vec![text.into()])
}

fn call_tool_err(message: &str) -> CallToolResult {
    CallToolResult::text_content(vec![format!("Error: {message}").into()])
}

fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn parse_hex_input(input: &str) -> Result<Vec<u8>, String> {
    let mut cleaned = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '0' && matches!(chars.peek(), Some('x' | 'X')) {
            chars.next();
            continue;
        }
        if ch.is_ascii_hexdigit() {
            cleaned.push(ch);
        }
    }

    if cleaned.is_empty() {
        return Ok(Vec::new());
    }
    if !cleaned.len().is_multiple_of(2) {
        return Err("hex input has odd number of digits".to_owned());
    }

    let mut out = Vec::with_capacity(cleaned.len() / 2);
    for chunk in cleaned.as_bytes().chunks_exact(2) {
        let s = std::str::from_utf8(chunk).map_err(|e| e.to_string())?;
        let byte = u8::from_str_radix(s, 16).map_err(|e| e.to_string())?;
        out.push(byte);
    }
    Ok(out)
}

async fn run() -> Result<(), String> {
    let handler = MirHandler::default();
    let server_details = InitializeResult {
        server_info: Implementation {
            name: "kajit".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            description: Some(
                "Kajit MCP server: interpreter debugging plus persistent lockstep LLDB sessions."
                    .into(),
            ),
            title: Some("Kajit Debugger".into()),
            icons: vec![],
            website_url: None,
        },
        capabilities: ServerCapabilities {
            tools: Some(ServerCapabilitiesTools {
                list_changed: Some(false),
            }),
            ..Default::default()
        },
        protocol_version: LATEST_PROTOCOL_VERSION.into(),
        instructions: Some(
            "Kajit MCP server. Use `session_*` tools for reversible CFG-MIR interpreter debugging and `debug_session_*` tools for persistent LLDB-backed lockstep sessions."
                .into(),
        ),
        meta: None,
    };

    let transport = StdioTransport::new(TransportOptions::default())
        .map_err(|e| format!("failed to create stdio transport: {e:?}"))?;
    let options = McpServerOptions {
        server_details,
        transport,
        handler: handler.to_mcp_server_handler(),
        task_store: None,
        client_task_store: None,
    };

    let server = server_runtime::create_server(options);
    server
        .start()
        .await
        .map_err(|e| format!("MCP server error: {e:?}"))?;
    Ok(())
}

/// Run the MCP server (real mode — handles MCP protocol directly).
pub async fn run_real() -> Result<(), String> {
    run().await
}

/// Run the MCP proxy (spawns --real subprocess, forwards stdin/stdout).
pub async fn run_mcp_proxy() -> Result<(), String> {
    run_proxy().await
}

/// Proxy mode: spawn the real MCP server as a subprocess and forward
/// stdin/stdout bidirectionally. This lets the MCP connection survive
/// rebuilds — just call the `reload` tool to restart the subprocess.
async fn run_proxy() -> Result<(), String> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::Command;

    let exe = std::env::current_exe().map_err(|e| format!("can't find self: {e}"))?;

    loop {
        let mut child = Command::new(&exe)
            .args(["mcp", "--real"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| format!("failed to spawn --real: {e}"))?;

        let mut child_stdin = child.stdin.take().unwrap();
        let child_stdout = child.stdout.take().unwrap();

        let mut proxy_stdin = BufReader::new(tokio::io::stdin());
        let mut child_stdout_reader = BufReader::new(child_stdout);

        // Forward proxy stdin → child stdin, child stdout → proxy stdout
        let mut proxy_stdout = tokio::io::stdout();

        #[allow(unused_assignments)]
        let mut should_reload = false;

        loop {
            let mut from_client = String::new();
            let mut from_child = String::new();

            tokio::select! {
                result = proxy_stdin.read_line(&mut from_client) => {
                    match result {
                        Ok(0) => return Ok(()), // EOF from client
                        Ok(_) => {
                            // Check if this is a reload request
                            if from_client.contains("\"reload\"") && from_client.contains("tools/call") {
                                // Parse JSON-RPC to extract the ID, send back a success response
                                if let Ok(parsed) = serde_json::from_str::<JsonValue>(&from_client) {
                                    let id = parsed.get("id").cloned().unwrap_or(JsonValue::Null);
                                    let response = json!({
                                        "jsonrpc": "2.0",
                                        "id": id,
                                        "result": {
                                            "content": [{
                                                "type": "text",
                                                "text": "Reloading backend..."
                                            }]
                                        }
                                    });
                                    let resp_str = serde_json::to_string(&response).unwrap();
                                    proxy_stdout.write_all(resp_str.as_bytes()).await.ok();
                                    proxy_stdout.write_all(b"\n").await.ok();
                                    proxy_stdout.flush().await.ok();
                                }
                                should_reload = true;
                                break;
                            }
                            // Forward to child
                            child_stdin.write_all(from_client.as_bytes()).await.ok();
                            child_stdin.flush().await.ok();
                        }
                        Err(e) => return Err(format!("stdin read error: {e}")),
                    }
                }
                result = child_stdout_reader.read_line(&mut from_child) => {
                    match result {
                        Ok(0) => {
                            // Child died — restart
                            should_reload = true;
                            break;
                        }
                        Ok(_) => {
                            // Forward to client
                            proxy_stdout.write_all(from_child.as_bytes()).await.ok();
                            proxy_stdout.flush().await.ok();
                        }
                        Err(e) => return Err(format!("child stdout read error: {e}")),
                    }
                }
            }
        }

        // Kill child and restart if reload requested
        child.kill().await.ok();
        child.wait().await.ok();

        if !should_reload {
            return Ok(());
        }
        eprintln!("[kajit-mcp] reloading backend...");
    }
}

#[cfg(test)]
mod tests {
    use super::{MirTools, encode_hex, parse_hex_input};

    #[test]
    fn parse_hex_accepts_common_formats() {
        assert_eq!(parse_hex_input("8101").unwrap(), vec![0x81, 0x01]);
        assert_eq!(parse_hex_input("81 01").unwrap(), vec![0x81, 0x01]);
        assert_eq!(parse_hex_input("[0x81, 0x01]").unwrap(), vec![0x81, 0x01]);
    }

    #[test]
    fn parse_hex_rejects_odd_digits() {
        assert!(parse_hex_input("abc").is_err());
    }

    #[test]
    fn encode_hex_roundtrip() {
        let bytes = vec![0x00, 0x7f, 0x80, 0xff];
        let encoded = encode_hex(&bytes);
        let decoded = parse_hex_input(&encoded).unwrap();
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn tool_schema_property_keys_follow_client_constraints() {
        let tools = MirTools::tools();
        let value = serde_json::to_value(tools).unwrap();
        let arr = value.as_array().expect("tools should serialize as array");

        for tool in arr {
            let props = tool
                .get("inputSchema")
                .and_then(|schema| schema.get("properties"))
                .and_then(|props| props.as_object())
                .expect("tool should have inputSchema.properties");
            for key in props.keys() {
                assert!(!key.is_empty(), "property key must not be empty");
                assert!(key.len() <= 64, "property key too long: {key}");
                assert!(
                    key.chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-')),
                    "property key has unsupported characters: {key}"
                );
            }
        }
    }
}
