#![allow(clippy::enum_variant_names)]

use std::collections::HashMap;
#[cfg(feature = "lldb")]
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
#[cfg(feature = "lldb")]
use kajit::lockstep::{JitDebugger, LockstepSession, LockstepSessionStatus};
#[cfg(feature = "lldb")]
use kajit_hir_text::parse_hir;
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
    /// Optional path to handwritten HIR text. When provided, this is compiled instead of a shape-based decoder.
    #[serde(default)]
    hir_path: Option<String>,
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
    name = "debug_session_cfg_context",
    description = "Show CFG-MIR block and edge context for the current or specified block in a persistent lockstep debug session."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionCfgContextTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
    /// Optional block id to inspect; defaults to the interpreter's current block.
    #[serde(default)]
    block_id: Option<u64>,
}

#[mcp_tool(
    name = "debug_session_vregs",
    description = "Inspect interpreter values, static homes, live homes, and current owners for selected vregs at the current stop."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct DebugSessionVregsTool {
    /// Session identifier returned by debug_session_new.
    session_id: u64,
    /// Optional comma-separated vreg list (e.g. '338,344' or 'v338,v344').
    /// Defaults to the current op's def/use vregs.
    #[serde(default)]
    vregs: Option<String>,
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
        DebugSessionCfgContextTool,
        DebugSessionVregsTool,
        DebugSessionLldbTool
    ]
);

struct ServerState {
    sessions: HashMap<u64, DebuggerSession>,
    next_session_id: u64,
    #[cfg(feature = "lldb")]
    debug_sessions: HashMap<u64, DebugSessionHandle>,
    #[cfg(feature = "lldb")]
    next_debug_session_id: u64,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            sessions: HashMap::new(),
            next_session_id: 1,
            #[cfg(feature = "lldb")]
            debug_sessions: HashMap::new(),
            #[cfg(feature = "lldb")]
            next_debug_session_id: 1,
        }
    }
}

/// Command sent from MCP handler to the debug session worker thread.
#[cfg(feature = "lldb")]
enum DebugCommand {
    Step { count: usize },
    State,
    Disassemble { context: usize },
    Registers { names: Vec<String> },
    Memory { address: u64, len: usize },
    Backtrace,
    SourceInfo,
    CfgContext { block_id: Option<u64> },
    Vregs { vregs: Vec<u32> },
    Lldb { command: String },
    Close,
}

/// Handle to a debug session running on a dedicated worker thread.
/// The LLDB types are !Send, so they live entirely on the worker thread.
/// Communication is via channels.
#[cfg(feature = "lldb")]
struct DebugSessionHandle {
    cmd_tx: std::sync::mpsc::Sender<DebugCommand>,
    resp_rx: std::sync::mpsc::Receiver<Result<JsonValue, String>>,
}

#[cfg(feature = "lldb")]
impl DebugSessionHandle {
    /// Send a command and wait for the response with a timeout.
    fn call(&self, cmd: DebugCommand, timeout: std::time::Duration) -> Result<JsonValue, String> {
        self.cmd_tx
            .send(cmd)
            .map_err(|_| "debug session worker thread has exited".to_owned())?;
        self.resp_rx.recv_timeout(timeout).map_err(|e| match e {
            std::sync::mpsc::RecvTimeoutError::Timeout => "debug session timed out".to_owned(),
            std::sync::mpsc::RecvTimeoutError::Disconnected => {
                "debug session worker thread has exited".to_owned()
            }
        })?
    }
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
        tracing::info!(tool = name, "call_tool");
        let result = self.call_tool_inner(name, args);
        match &result {
            Ok(_) => tracing::info!(tool = name, "call_tool OK"),
            Err(e) => tracing::error!(tool = name, error = %e, "call_tool ERR"),
        }
        result
    }

    fn call_tool_inner(
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
            "debug_session_cfg_context" => self.debug_session_cfg_context(args),
            "debug_session_vregs" => self.debug_session_vregs(args),
            "debug_session_lldb" => self.debug_session_lldb(args),
            other => Err(format!("unknown tool: {other}")),
        }
    }

    fn session_new(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        tracing::info!("session_new: creating interpreter session");
        let mir_path = arg_str(args, "cfg_mir_path")?;
        let mir_text = std::fs::read_to_string(&mir_path)
            .map_err(|e| format!("failed to read {mir_path}: {e}"))?;
        let input_hex = arg_opt_str(args, "input_hex").unwrap_or_default();
        let input = parse_hex_input(&input_hex)?;
        let program = kajit_mir_text::parse_cfg_mir(&mir_text).map_err(|e| e.to_string())?;
        let args = kajit_types::Arguments::new();
        let session = DebuggerSession::new(&program, &input, &args).map_err(|e| e.to_string())?;

        let mut state = self.lock_state()?;
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
        md.push_str("\n**Final state:**\n");
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
            let hir_path = arg_opt_str(args, "hir_path");

            // Allocate session ID first
            let session_id = {
                let mut state = self.lock_state()?;
                let id = state.next_debug_session_id;
                state.next_debug_session_id += 1;
                id
            };

            // Do compilation + harness generation on this thread (pure Rust, no LLDB)
            let prepared =
                DebugDiffSession::prepare(&format, &ty, &input_hex, hir_path.as_deref())?;

            // Spawn worker thread that owns the LLDB session
            let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
            let (resp_tx, resp_rx) = std::sync::mpsc::channel();

            let init_resp_tx = resp_tx.clone();
            std::thread::spawn(move || {
                // Launch LLDB on the worker thread (LLDB types are !Send)
                match DebugDiffSession::launch(prepared) {
                    Ok(session) => {
                        let snapshot = session.snapshot_markdown(session_id);
                        let _ = init_resp_tx.send(Ok(json!({ "text": snapshot })));
                        debug_session_worker(session_id, session, cmd_rx, resp_tx);
                    }
                    Err(e) => {
                        let _ = init_resp_tx.send(Err(e));
                    }
                }
            });

            // Wait for the worker to finish initialization (with timeout)
            let init_result = resp_rx
                .recv_timeout(std::time::Duration::from_secs(60))
                .map_err(|e| match e {
                    std::sync::mpsc::RecvTimeoutError::Timeout => {
                        "debug session creation timed out after 60s".to_owned()
                    }
                    std::sync::mpsc::RecvTimeoutError::Disconnected => {
                        "debug session worker thread exited during init".to_owned()
                    }
                })??;

            let handle = DebugSessionHandle { cmd_tx, resp_rx };
            self.lock_state()?.debug_sessions.insert(session_id, handle);

            Ok(init_result)
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
            if let Some(handle) = state.debug_sessions.remove(&session_id) {
                // Tell the worker to shut down; ignore errors (it may have already exited)
                let _ = handle.cmd_tx.send(DebugCommand::Close);
                Ok(json!({ "session_id": session_id, "closed": true }))
            } else {
                Ok(json!({ "session_id": session_id, "closed": false }))
            }
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
            self.debug_session_call(
                session_id,
                DebugCommand::Step { count },
                std::time::Duration::from_secs(30),
            )
        }
    }

    /// Helper: look up a debug session handle and send a command with timeout.
    #[cfg(feature = "lldb")]
    fn debug_session_call(
        &self,
        session_id: u64,
        cmd: DebugCommand,
        timeout: std::time::Duration,
    ) -> Result<JsonValue, String> {
        let state = self.lock_state()?;
        let handle = state
            .debug_sessions
            .get(&session_id)
            .ok_or_else(|| format!("unknown session_id: {session_id}"))?;
        handle.call(cmd, timeout)
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
            self.debug_session_call(
                session_id,
                DebugCommand::State,
                std::time::Duration::from_secs(10),
            )
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
            self.debug_session_call(
                session_id,
                DebugCommand::Disassemble { context },
                std::time::Duration::from_secs(10),
            )
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
            self.debug_session_call(
                session_id,
                DebugCommand::Registers { names },
                std::time::Duration::from_secs(10),
            )
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
            self.debug_session_call(
                session_id,
                DebugCommand::Memory { address, len },
                std::time::Duration::from_secs(10),
            )
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
            self.debug_session_call(
                session_id,
                DebugCommand::Backtrace,
                std::time::Duration::from_secs(10),
            )
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
            self.debug_session_call(
                session_id,
                DebugCommand::SourceInfo,
                std::time::Duration::from_secs(10),
            )
        }
    }

    fn debug_session_cfg_context(
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
            let block_id = arg_opt_u64(args, "block_id");
            self.debug_session_call(
                session_id,
                DebugCommand::CfgContext { block_id },
                std::time::Duration::from_secs(10),
            )
        }
    }

    fn debug_session_vregs(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        #[cfg(not(feature = "lldb"))]
        {
            let _ = args;
            Err("debug sessions require kajit to be built with the `lldb` feature".to_owned())
        }

        #[cfg(feature = "lldb")]
        {
            let session_id = arg_u64(args, "session_id")?;
            let vregs = parse_vreg_list(arg_opt_str(args, "vregs").as_deref())?;
            self.debug_session_call(
                session_id,
                DebugCommand::Vregs { vregs },
                std::time::Duration::from_secs(10),
            )
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
            self.debug_session_call(
                session_id,
                DebugCommand::Lldb { command },
                std::time::Duration::from_secs(10),
            )
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

/// Intermediate data from compilation + harness generation (pure Rust, no LLDB).
/// This is `Send` and gets moved to the worker thread for LLDB launch.
#[cfg(feature = "lldb")]
struct PreparedSession {
    format: String,
    ty: String,
    input_hex: String,
    input: Vec<u8>,
    exe_path: PathBuf,
    listing_path: PathBuf,
    artifacts: kajit::PipelineArtifacts,
}

#[cfg(feature = "lldb")]
struct DebugDiffSession {
    format: String,
    ty: String,
    input: Vec<u8>,
    exe_path: PathBuf,
    listing_path: PathBuf,
    lockstep: LockstepSession<LldbJitDebugger>,
}

#[cfg(feature = "lldb")]
impl DebugDiffSession {
    fn output_size_from_hir_module(module: &kajit_hir::Module) -> Result<usize, String> {
        use kajit_hir::Type;

        fn type_size(module: &kajit_hir::Module, ty: &Type) -> Result<usize, String> {
            Ok(match ty {
                Type::Unit => 0,
                Type::Bool => 1,
                Type::Integer(int) => usize::from(int.bits / 8),
                Type::Ref { .. } | Type::Address { .. } | Type::Handle { .. } => 8,
                Type::Slice { .. } | Type::Str { .. } => 16,
                Type::Array { element, len } => type_size(module, element)? * *len,
                Type::Named { def, .. } => module.type_defs[*def]
                    .size
                    .map(|size| size as usize)
                    .ok_or_else(|| {
                        format!(
                            "destination type '{}' is missing size metadata",
                            module.type_defs[*def].name
                        )
                    })?,
            })
        }

        let (_, function) = module
            .functions
            .iter()
            .next()
            .ok_or_else(|| "HIR module should contain at least one function".to_owned())?;
        // The second parameter is the output pointer.
        let out_param = function.params.get(1).ok_or_else(|| {
            "debug sessions require a function with an output parameter".to_owned()
        })?;
        // If out is a reference/pointer, use the pointee type size
        let out_ty = match &out_param.ty {
            Type::Ref { pointee, .. } => pointee.as_ref(),
            other => other,
        };
        type_size(module, out_ty)
    }

    /// Compile the pipeline and generate the harness binary. This is pure Rust
    /// (no LLDB), so the result is `Send` and can be moved to a worker thread.
    fn prepare(
        format: &str,
        ty: &str,
        input_hex: &str,
        hir_path: Option<&str>,
    ) -> Result<PreparedSession, String> {
        tracing::info!(
            format,
            ty,
            input_hex,
            ?hir_path,
            "debug_session: prepare start"
        );
        let pipeline_opts = kajit::PipelineOptions::from_env();
        let (artifacts, output_size) = if let Some(hir_path) = hir_path {
            let hir_text = std::fs::read_to_string(hir_path)
                .map_err(|e| format!("failed to read HIR file '{hir_path}': {e}"))?;
            let module = parse_hir(&hir_text).map_err(|e| format!("HIR parse error: {e}"))?;
            let registry = kajit::ir::IntrinsicRegistry::empty();
            let output_size = Self::output_size_from_hir_module(&module)?;
            (
                kajit::compile_pipeline_from_hir_module(&module, &registry, &pipeline_opts),
                output_size,
            )
        } else {
            let kind = match format {
                "postcard" => kajit::DecoderKind::Postcard,
                other => return Err(format!("unknown format '{other}', expected 'postcard'")),
            };

            let shape = super::try_resolve_shape(ty)?;
            tracing::info!(ty, "debug_session: compiling pipeline");
            let artifacts = kajit::compile_pipeline(shape, kind, &pipeline_opts);
            tracing::info!("debug_session: pipeline compiled");
            let output_size = shape.layout.sized_layout().map(|l| l.size()).unwrap_or(0);
            (artifacts, output_size)
        };

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
            alloc_map: Some(&artifacts.alloc_map),
            intrinsic_calls,
            extern_addr_relocs: artifacts.extern_addr_relocs.clone(),
        };

        let exe_path = kajit::harness::generate_harness(&harness_input, &output_dir, &base_name)
            .map_err(|e| format!("error generating harness: {e}"))?;

        tracing::info!(harness = %exe_path.display(), output_size, listing = %listing_path.display(), "debug_session: harness generated");

        let metadata = std::fs::metadata(&exe_path)
            .map_err(|e| format!("harness binary not found at {}: {e}", exe_path.display()))?;
        tracing::info!(
            binary_size = metadata.len(),
            "debug_session: binary verified"
        );

        let input = parse_hex_input(input_hex)?;

        Ok(PreparedSession {
            format: format.to_owned(),
            ty: ty.to_owned(),
            input_hex: input_hex.to_owned(),
            input,
            exe_path,
            listing_path,
            artifacts,
        })
    }

    /// Launch LLDB and create the lockstep session. Must be called on the
    /// thread that will own the session (LLDB types are !Send).
    fn launch(prepared: PreparedSession) -> Result<Self, String> {
        tracing::info!(input_hex = %prepared.input_hex, "debug_session: launching LLDB");
        let debugger =
            LldbJitDebugger::launch(prepared.exe_path.to_str().unwrap(), &prepared.input_hex)
                .map_err(|e| format!("error launching LLDB: {e}"))?;
        tracing::info!("debug_session: LLDB launched, creating lockstep session");
        let lockstep = LockstepSession::new(
            &prepared.artifacts.cfg_program,
            &prepared.input,
            &prepared.artifacts.location_map,
            prepared.artifacts.decoder.cfg_mir_lines().to_vec(),
            prepared.artifacts.backend_debug_info.as_ref(),
            prepared.artifacts.decoder.entry_offset(),
            debugger,
        )
        .map_err(|e| e.to_string())?;
        tracing::info!("debug_session: session ready");

        Ok(Self {
            format: prepared.format,
            ty: prepared.ty,
            input: prepared.input,
            exe_path: prepared.exe_path,
            listing_path: prepared.listing_path,
            lockstep,
        })
    }

    fn is_running(&self) -> bool {
        self.lockstep.is_running()
    }

    fn current_line_text(&self, line: u32) -> String {
        self.lockstep.current_line_text(line)
    }

    fn dump_current_op_vregs(&self) -> Option<String> {
        use kajit::lockstep::op_def_uses_and_kind;
        let func = &self.lockstep.cfg_program.funcs[0];
        let loc = &self.lockstep.interpreter.state().location;
        let (_def, uses, _) = op_def_uses_and_kind(func, loc);
        if uses.is_empty() {
            return None;
        }
        let state = self.lockstep.interpreter.state();
        let mut parts = Vec::new();
        for vreg in &uses {
            let iv = if vreg.index() < state.vregs.len() {
                state.vregs[vreg.index()]
            } else {
                0
            };
            let jit_val = self
                .lockstep
                .debugger
                .read_register(
                    self.lockstep
                        .location_tracker
                        .location_for(&self.lockstep.location_map, vreg.index() as u32)
                        .and_then(|loc| match loc {
                            kajit::harness::VRegLocation::Register(p) => Some(p),
                            _ => None,
                        })
                        .unwrap_or(255),
                )
                .ok();
            parts.push(format!(
                "v{}=interp:{}{}",
                vreg.index(),
                iv,
                jit_val.map(|j| format!(",jit:{j}")).unwrap_or_default()
            ));
        }
        Some(parts.join(" | "))
    }

    fn step_forward(&mut self) -> Result<String, String> {
        self.lockstep.step_forward().map_err(|e| e.to_string())
    }

    fn snapshot_markdown(&self, session_id: u64) -> String {
        let status = match self.lockstep.status {
            LockstepSessionStatus::Running => "running",
            LockstepSessionStatus::Diverged => "diverged",
            LockstepSessionStatus::Completed => "completed",
        };
        let pc = self.lockstep.debugger.read_pc().ok();
        let dwarf_line = self.lockstep.current_mapped_line();
        let mut out = format!(
            "**Debug session {}** `{}` `{}`\n\nstatus: **{}** | steps={} | input=`{}`\nexe: `{}`\nlisting: `{}`\n",
            session_id,
            self.format,
            self.ty,
            status,
            self.lockstep.jit_steps,
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
        let state = self.lockstep.interpreter.state();
        out.push_str(&format_state_markdown(&state));
        if let Some(provenance) = self.current_provenance_markdown(&state) {
            out.push('\n');
            out.push_str("**provenance**\n");
            out.push_str(&provenance);
        }
        if let Some(div) = &self.lockstep.divergence {
            out.push_str("\n**last divergence**\n");
            out.push_str(&div.source_line);
            out.push('\n');
        }
        out
    }

    fn current_provenance_markdown(&self, state: &DebuggerState) -> Option<String> {
        let loc = &state.location;
        let func = self.lockstep.cfg_program.funcs.first()?;
        let block = func.blocks.get(loc.block.index())?;
        let op_id = if loc.at_terminator {
            kajit_mir::cfg_mir::OpId::Term(block.term)
        } else {
            kajit_mir::cfg_mir::OpId::Inst(*block.insts.get(loc.next_inst_index)?)
        };
        let cfg_line = kajit::lockstep::loc_to_line(&self.lockstep.op_to_line, loc);
        let mut out = String::new();
        out.push_str(&format!(
            "cfg_line={} | op=`{}`\n",
            cfg_line,
            self.current_line_text(cfg_line)
        ));

        if let Some(scope_id) = self
            .lockstep
            .cfg_program
            .op_debug_scope(func.lambda_id, op_id)
        {
            out.push_str(&format!(
                "op_scope: {}\n",
                self.format_scope_chain(scope_id)
            ));
        }
        if let Some(value_id) = self
            .lockstep
            .cfg_program
            .op_debug_value(func.lambda_id, op_id)
        {
            out.push_str(&format!(
                "op_value: {}\n",
                self.format_debug_value(value_id)
            ));
        }

        let (def_vreg, use_vregs, _) = kajit::lockstep::op_def_uses_and_kind(func, loc);
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
        if let Some(scope_id) = self.lockstep.cfg_program.vreg_debug_scope(vreg) {
            line.push_str(&format!(" | scope {}", self.format_scope_chain(scope_id)));
        }
        if let Some(value_id) = self.lockstep.cfg_program.vreg_debug_value(vreg) {
            line.push_str(&format!(" | value {}", self.format_debug_value(value_id)));
        }
        line
    }

    fn format_scope_chain(&self, scope_id: kajit_ir::DebugScopeId) -> String {
        let mut chain = Vec::new();
        let mut current = Some(scope_id);
        while let Some(id) = current {
            let scope = &self.lockstep.cfg_program.debug.scopes[id];
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
        let value = &self.lockstep.cfg_program.debug.values[value_id];
        match &value.kind {
            kajit_ir::DebugValueKind::Field { offset } => {
                format!("{} [field offset={}]", value.name, offset)
            }
            kajit_ir::DebugValueKind::Named => format!("{} [named]", value.name),
        }
    }
}

/// Worker thread main loop: owns a DebugDiffSession and processes commands.
#[cfg(feature = "lldb")]
fn debug_session_worker(
    session_id: u64,
    mut session: DebugDiffSession,
    cmd_rx: std::sync::mpsc::Receiver<DebugCommand>,
    resp_tx: std::sync::mpsc::Sender<Result<JsonValue, String>>,
) {
    tracing::info!(session_id, "debug worker: started");
    while let Ok(cmd) = cmd_rx.recv() {
        let result = match cmd {
            DebugCommand::Close => {
                tracing::info!(session_id, "debug worker: closing");
                break;
            }
            DebugCommand::Step { count } => {
                worker_handle_step(&mut session, session_id, count)
            }
            DebugCommand::State => {
                Ok(json!({ "text": session.snapshot_markdown(session_id) }))
            }
            DebugCommand::Disassemble { context } => session
                .lockstep
                .debugger
                .disassemble_around_pc(context)
                .map(|text| json!({ "text": text }))
                .map_err(|e| e.to_string()),
            DebugCommand::Registers { names } => {
                worker_handle_registers(&session, &names)
            }
            DebugCommand::Memory { address, len } => session
                .lockstep
                .debugger
                .read_memory(address, len)
                .map(|bytes| {
                    json!({
                        "text": format!("mem[0x{address:x}..0x{:x}] = `{}`", address + bytes.len() as u64, encode_hex(&bytes)),
                        "address": address,
                        "len": bytes.len(),
                        "hex": encode_hex(&bytes),
                    })
                })
                .map_err(|e| e.to_string()),
            DebugCommand::Backtrace => session
                .lockstep
                .debugger
                .backtrace()
                .map(|text| json!({ "text": text }))
                .map_err(|e| e.to_string()),
            DebugCommand::SourceInfo => {
                worker_handle_source_info(&session)
            }
            DebugCommand::CfgContext { block_id } => {
                worker_handle_cfg_context(&session, session_id, block_id)
            }
            DebugCommand::Vregs { vregs } => {
                worker_handle_vregs(&session, session_id, &vregs)
            }
            DebugCommand::Lldb { command } => session
                .lockstep
                .debugger
                .execute_command(&command)
                .map(|text| json!({ "text": text }))
                .map_err(|e| e.to_string()),
        };
        if resp_tx.send(result).is_err() {
            // MCP handler disconnected (e.g. timeout)
            tracing::warn!(session_id, "debug worker: response channel closed, exiting");
            break;
        }
    }
    tracing::info!(session_id, "debug worker: exited");
}

#[cfg(feature = "lldb")]
fn worker_handle_step(
    session: &mut DebugDiffSession,
    session_id: u64,
    count: usize,
) -> Result<JsonValue, String> {
    let log_path = format!("/tmp/kajit-lockstep-{session_id}.log");
    let _ = std::fs::write(&log_path, "");

    let mut out = format!("**Debug session {}**\n\n", session_id);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    for i in 0..count {
        if i > 0 && std::time::Instant::now() > deadline {
            let msg = format!("TIMEOUT: completed {i}/{count} steps in 5s, stopping early\n");
            let _ = append_log(&log_path, &msg);
            out.push_str(&msg);
            break;
        }
        let cur_line = session.lockstep.current_mapped_line();
        let cur_text = session.current_line_text(cur_line);
        let mut step_log = format!(
            ">>> step {} | about to execute line {}: {}",
            i + 1,
            cur_line,
            cur_text.trim()
        );
        if let Some(vreg_info) = session.dump_current_op_vregs() {
            step_log.push_str(&format!("\n    {}", vreg_info));
        }
        let _ = append_log(&log_path, &step_log);
        match session.step_forward() {
            Ok(step) => {
                let _ = append_log(&log_path, &step);
                out.push_str(&step);
                out.push('\n');
                if !session.is_running() {
                    break;
                }
            }
            Err(e) => {
                out.push_str(&format!("ERROR: {e}\n"));
                break;
            }
        }
    }
    out.push('\n');
    out.push_str(&session.snapshot_markdown(session_id));
    Ok(json!({ "text": out }))
}

#[cfg(feature = "lldb")]
fn worker_handle_registers(
    session: &DebugDiffSession,
    names: &[String],
) -> Result<JsonValue, String> {
    let mut values = serde_json::Map::new();
    let mut text = String::new();
    for name in names {
        let value = session
            .lockstep
            .debugger
            .read_register_by_name(name)
            .map_err(|e| e.to_string())?;
        values.insert(name.clone(), json!(value));
        text.push_str(&format!("{name}=0x{value:x} ({value})\n"));
    }
    Ok(json!({ "text": text, "registers": values }))
}

#[cfg(feature = "lldb")]
fn worker_handle_source_info(session: &DebugDiffSession) -> Result<JsonValue, String> {
    let pc = session
        .lockstep
        .debugger
        .read_pc()
        .map_err(|e| e.to_string())?;
    let dwarf_line = session.lockstep.current_mapped_line();
    let cfg_line = session.current_line_text(dwarf_line);
    let lldb = session
        .lockstep
        .debugger
        .source_info()
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "text": format!("pc=0x{pc:x}\ndwarf_line={dwarf_line}\ncfg=`{cfg_line}`\n\n{lldb}"),
        "pc": pc,
        "dwarf_line": dwarf_line,
        "cfg_line": cfg_line,
    }))
}

#[cfg(feature = "lldb")]
fn worker_handle_cfg_context(
    session: &DebugDiffSession,
    _session_id: u64,
    requested_block_id: Option<u64>,
) -> Result<JsonValue, String> {
    let func = session
        .lockstep
        .cfg_program
        .funcs
        .first()
        .ok_or_else(|| "debug session has no function".to_owned())?;
    let current_loc = session.lockstep.interpreter.state().location;
    let block_id = requested_block_id.unwrap_or(current_loc.block.index() as u64) as usize;
    let block = func
        .blocks
        .get(block_id)
        .ok_or_else(|| format!("unknown block_id: b{block_id}"))?;

    let mut text = String::new();
    text.push_str(&format!(
        "block b{}{}\n",
        block.id.index(),
        if block.id == current_loc.block {
            " (current)"
        } else {
            ""
        }
    ));
    text.push_str(&format!(
        "params: {}\n",
        format_cfg_vregs(
            &block.params,
            &session.lockstep.interpreter.state(),
            Some(&session.lockstep),
        )
    ));
    text.push_str(&format!("dead: {}\n", block.dead));

    if block.id == current_loc.block {
        let (def_vreg, use_vregs, _) = kajit::lockstep::op_def_uses_and_kind(func, &current_loc);
        if def_vreg.is_some() || !use_vregs.is_empty() {
            text.push_str("current_op_locations:\n");
            if let Some(vreg) = def_vreg {
                text.push_str(&format!(
                    "  def v{} -> {}\n",
                    vreg.index(),
                    format_live_location(&session.lockstep, vreg.index() as u32)
                ));
            }
            for vreg in use_vregs {
                text.push_str(&format!(
                    "  use v{} -> {}\n",
                    vreg.index(),
                    format_live_location(&session.lockstep, vreg.index() as u32)
                ));
            }
        }
    }

    text.push_str("insts:\n");
    for &inst_id in &block.insts {
        let inst = &func.insts[inst_id.index()];
        let loc = kajit_mir::ProgramLocation {
            block: block.id,
            next_inst_index: block
                .insts
                .iter()
                .position(|&candidate| candidate == inst_id)
                .unwrap_or(0),
            at_terminator: false,
        };
        let line = kajit::lockstep::loc_to_line(&session.lockstep.op_to_line, &loc);
        text.push_str(&format!("  line {line}: {:?}\n", inst.op));
    }

    let term_loc = kajit_mir::ProgramLocation {
        block: block.id,
        next_inst_index: 0,
        at_terminator: true,
    };
    let term_line = kajit::lockstep::loc_to_line(&session.lockstep.op_to_line, &term_loc);
    text.push_str(&format!(
        "term: line {}: {:?}\n",
        term_line,
        func.terms[block.term.index()]
    ));

    text.push_str("preds:\n");
    for &edge_id in &block.preds {
        let edge = &func.edges[edge_id.index()];
        text.push_str(&format!(
            "  e{}: b{} -> b{}{}\n",
            edge.id.index(),
            edge.from.index(),
            edge.to.index(),
            format_edge_args(
                edge,
                &session.lockstep.interpreter.state(),
                Some(&session.lockstep),
            )
        ));
    }

    text.push_str("succs:\n");
    for &edge_id in &block.succs {
        let edge = &func.edges[edge_id.index()];
        text.push_str(&format!(
            "  e{}: b{} -> b{}{}\n",
            edge.id.index(),
            edge.from.index(),
            edge.to.index(),
            format_edge_args(
                edge,
                &session.lockstep.interpreter.state(),
                Some(&session.lockstep),
            )
        ));
    }

    Ok(json!({
        "text": text,
        "block_id": block.id.index(),
    }))
}

#[cfg(feature = "lldb")]
fn worker_handle_vregs(
    session: &DebugDiffSession,
    _session_id: u64,
    requested: &[u32],
) -> Result<JsonValue, String> {
    let func = session
        .lockstep
        .cfg_program
        .funcs
        .first()
        .ok_or_else(|| "debug session has no function".to_owned())?;
    let current_loc = session.lockstep.interpreter.state().location;
    let current_state = session.lockstep.interpreter.state();
    let mut vregs = if requested.is_empty() {
        let (def_vreg, use_vregs, _) = kajit::lockstep::op_def_uses_and_kind(func, &current_loc);
        let mut derived = Vec::new();
        if let Some(def_vreg) = def_vreg {
            derived.push(def_vreg.index() as u32);
        }
        for vreg in use_vregs {
            let idx = vreg.index() as u32;
            if !derived.contains(&idx) {
                derived.push(idx);
            }
        }
        derived
    } else {
        requested.to_vec()
    };
    if vregs.is_empty() {
        return Err("no vregs specified and current op has no def/use vregs".to_owned());
    }
    vregs.sort_unstable();

    let mut text = String::new();
    let mut rows = Vec::new();
    for vreg_idx in vregs {
        let value = current_state
            .vregs
            .get(vreg_idx as usize)
            .copied()
            .unwrap_or(0);
        let static_loc = format_static_location(&session.lockstep, vreg_idx);
        let live_loc = format_live_location(&session.lockstep, vreg_idx);
        let owner = session
            .lockstep
            .location_tracker
            .owner_of_vreg_location(&session.lockstep.location_map, vreg_idx);
        let owner_text = match owner {
            Some(owner_idx) if owner_idx == vreg_idx => "self".to_owned(),
            Some(owner_idx) => {
                let owner_val = current_state
                    .vregs
                    .get(owner_idx as usize)
                    .copied()
                    .unwrap_or(0);
                format!("v{owner_idx}={owner_val}")
            }
            None => "none".to_owned(),
        };
        text.push_str(&format!(
            "v{vreg_idx}={value}\n  static: {static_loc}\n  live: {live_loc}\n  owner: {owner_text}\n"
        ));
        rows.push(json!({
            "vreg": vreg_idx,
            "value": value,
            "static": static_loc,
            "live": live_loc,
            "owner": owner,
        }));
    }

    Ok(json!({ "text": text, "vregs": rows }))
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

#[cfg(feature = "lldb")]
fn format_cfg_vregs(
    vregs: &[kajit_ir::VReg],
    state: &DebuggerState,
    lockstep: Option<&LockstepSession<LldbJitDebugger>>,
) -> String {
    if vregs.is_empty() {
        return "[]".to_owned();
    }
    let parts: Vec<String> = vregs
        .iter()
        .map(|vreg| {
            let value = state.vregs.get(vreg.index()).copied().unwrap_or(0);
            match lockstep {
                Some(lockstep) => format!(
                    "v{}={} ({})",
                    vreg.index(),
                    value,
                    describe_location(lockstep, vreg.index() as u32)
                ),
                None => format!("v{}={value}", vreg.index()),
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

#[cfg(feature = "lldb")]
fn format_edge_args(
    edge: &kajit_mir::cfg_mir::Edge,
    state: &DebuggerState,
    lockstep: Option<&LockstepSession<LldbJitDebugger>>,
) -> String {
    if edge.args.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = edge
        .args
        .iter()
        .map(|arg| {
            let src_value = state.vregs.get(arg.source.index()).copied().unwrap_or(0);
            let dst_value = state.vregs.get(arg.target.index()).copied().unwrap_or(0);
            match lockstep {
                Some(lockstep) => format!(
                    "v{} <- v{} (src={}, dst={}, src_loc={}, dst_loc={})",
                    arg.target.index(),
                    arg.source.index(),
                    src_value,
                    dst_value,
                    describe_location(lockstep, arg.source.index() as u32),
                    describe_location(lockstep, arg.target.index() as u32),
                ),
                None => format!(
                    "v{} <- v{} (src={}, dst={})",
                    arg.target.index(),
                    arg.source.index(),
                    src_value,
                    dst_value
                ),
            }
        })
        .collect();
    format!(" args=[{}]", parts.join(", "))
}

#[cfg(feature = "lldb")]
fn format_live_location(lockstep: &LockstepSession<LldbJitDebugger>, vreg_index: u32) -> String {
    match lockstep
        .location_tracker
        .location_for(&lockstep.location_map, vreg_index)
    {
        Some(kajit::harness::VRegLocation::Register(preg)) => {
            format!("reg {}", kajit::harness::LocationMap::reg_name(preg))
        }
        Some(kajit::harness::VRegLocation::StackSlot(offset)) => format!("[sp+{offset}]"),
        Some(kajit::harness::VRegLocation::Constant(value)) => format!("const({value})"),
        None => "clobbered/unmapped".to_owned(),
    }
}

#[cfg(feature = "lldb")]
fn format_static_location(lockstep: &LockstepSession<LldbJitDebugger>, vreg_index: u32) -> String {
    match lockstep.location_map.static_locations.get(&vreg_index) {
        Some(kajit::harness::VRegLocation::Register(preg)) => {
            format!("reg {}", kajit::harness::LocationMap::reg_name(*preg))
        }
        Some(kajit::harness::VRegLocation::StackSlot(offset)) => format!("[sp+{offset}]"),
        Some(kajit::harness::VRegLocation::Constant(value)) => format!("const({value})"),
        None => "unallocated".to_owned(),
    }
}

#[cfg(feature = "lldb")]
fn describe_location(lockstep: &LockstepSession<LldbJitDebugger>, vreg_index: u32) -> String {
    let live = format_live_location(lockstep, vreg_index);
    let static_loc = format_static_location(lockstep, vreg_index);
    if live == static_loc {
        live
    } else {
        format!("live={live}, static={static_loc}")
    }
}

#[cfg(feature = "lldb")]
fn parse_vreg_list(spec: Option<&str>) -> Result<Vec<u32>, String> {
    let Some(spec) = spec else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    for raw in spec.split(',') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let digits = trimmed.strip_prefix('v').unwrap_or(trimmed);
        let idx = digits
            .parse::<u32>()
            .map_err(|_| format!("invalid vreg index `{trimmed}`"))?;
        if !result.contains(&idx) {
            result.push(idx);
        }
    }
    Ok(result)
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

fn append_log(path: &str, line: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")?;
    f.flush()
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
    let mut result = CallToolResult::text_content(vec![format!("Error: {message}").into()]);
    result.is_error = Some(true);
    result
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
            "Kajit MCP server. Use `session_*` tools for reversible CFG-MIR interpreter debugging and `debug_session_*` tools for persistent LLDB-backed lockstep sessions. The server logs to `/tmp/kajit-mcp.log` (set `RUST_LOG` for verbosity, default: info)."
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
    tracing::info!("kajit MCP server starting (real mode)");
    run().await
}

/// Run the MCP proxy (spawns --real subprocess, forwards stdin/stdout).
pub async fn run_mcp_proxy() -> Result<(), String> {
    run_proxy().await
}

fn proxy_reload_tool() -> JsonValue {
    json!({
        "name": "reload",
        "description": "Restart the Kajit MCP backend subprocess so the proxy picks up newly installed code.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }
    })
}

fn proxy_reload_success(id: JsonValue) -> JsonValue {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [{
                "type": "text",
                "text": "Reloading backend..."
            }]
        }
    })
}

fn proxy_tools_list_changed_notification() -> JsonValue {
    json!({
        "jsonrpc": "2.0",
        "method": "notifications/tools/list_changed"
    })
}

fn request_id_key(id: &JsonValue) -> Option<String> {
    serde_json::to_string(id).ok()
}

fn patch_initialize_response(response: &mut JsonValue) {
    let Some(result) = response.get_mut("result").and_then(|v| v.as_object_mut()) else {
        return;
    };
    let capabilities = result
        .entry("capabilities")
        .or_insert_with(|| json!({}))
        .as_object_mut();
    let Some(capabilities) = capabilities else {
        return;
    };
    let tools = capabilities
        .entry("tools")
        .or_insert_with(|| json!({}))
        .as_object_mut();
    let Some(tools) = tools else {
        return;
    };
    tools.insert("listChanged".to_owned(), JsonValue::Bool(true));
}

fn patch_tools_list_response(response: &mut JsonValue) {
    let Some(result) = response.get_mut("result").and_then(|v| v.as_object_mut()) else {
        return;
    };
    let tools = result
        .entry("tools")
        .or_insert_with(|| JsonValue::Array(Vec::new()))
        .as_array_mut();
    let Some(tools) = tools else {
        return;
    };
    let has_reload = tools
        .iter()
        .any(|tool| tool.get("name") == Some(&JsonValue::String("reload".to_owned())));
    if !has_reload {
        tools.push(proxy_reload_tool());
    }
}

/// Proxy mode: spawn the real MCP server as a subprocess and forward
/// stdin/stdout bidirectionally. This lets the MCP connection survive
/// rebuilds — just call the `reload` tool to restart the subprocess.
async fn run_proxy() -> Result<(), String> {
    use std::collections::{HashMap, HashSet};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::Command;

    let exe = std::env::current_exe().map_err(|e| format!("can't find self: {e}"))?;
    let mut pending_tool_list_ids = HashSet::<String>::new();
    let mut pending_initialize_ids = HashSet::<String>::new();
    // Track all in-flight request IDs so we can send errors if the child dies
    let mut inflight_requests = HashMap::<String, JsonValue>::new();

    loop {
        pending_tool_list_ids.clear();
        pending_initialize_ids.clear();
        inflight_requests.clear();

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
                            if let Ok(parsed) = serde_json::from_str::<JsonValue>(&from_client) {
                                // Track in-flight requests so we can send errors if child dies
                                if let Some(id) = parsed.get("id") {
                                    if let Some(key) = request_id_key(id) {
                                        inflight_requests.insert(key.clone(), id.clone());
                                    }
                                }

                                if let Some(method) = parsed.get("method").and_then(|v| v.as_str())
                                    && let Some(id_key) = parsed.get("id").and_then(request_id_key) {
                                        match method {
                                            "initialize" => {
                                                pending_initialize_ids.insert(id_key);
                                            }
                                            "tools/list" => {
                                                pending_tool_list_ids.insert(id_key);
                                            }
                                            "tools/call" => {
                                                let is_reload = parsed
                                                    .get("params")
                                                    .and_then(|params| params.get("name"))
                                                    .and_then(|name| name.as_str())
                                                    == Some("reload");
                                                if is_reload {
                                                    let id = parsed.get("id").cloned().unwrap_or(JsonValue::Null);
                                                    let response = proxy_reload_success(id);
                                                    let resp_str = serde_json::to_string(&response).unwrap();
                                                    proxy_stdout.write_all(resp_str.as_bytes()).await.ok();
                                                    proxy_stdout.write_all(b"\n").await.ok();
                                                    proxy_stdout.flush().await.ok();
                                                    should_reload = true;
                                                    break;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
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
                            // Child died — send error responses for all in-flight requests
                            for (_key, id) in inflight_requests.drain() {
                                let err_resp = json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "error": {
                                        "code": -32603,
                                        "message": "MCP backend process crashed"
                                    }
                                });
                                let s = serde_json::to_string(&err_resp).unwrap();
                                proxy_stdout.write_all(s.as_bytes()).await.ok();
                                proxy_stdout.write_all(b"\n").await.ok();
                            }
                            proxy_stdout.flush().await.ok();
                            should_reload = true;
                            break;
                        }
                        Ok(_) => {
                            let mut outbound = from_child.clone();
                            if let Ok(mut parsed) = serde_json::from_str::<JsonValue>(&from_child)
                                && let Some(id_key) = parsed.get("id").and_then(request_id_key) {
                                    // Response received — no longer in-flight
                                    inflight_requests.remove(&id_key);
                                    if pending_initialize_ids.remove(&id_key) {
                                        patch_initialize_response(&mut parsed);
                                        outbound = serde_json::to_string(&parsed)
                                            .map(|s| format!("{s}\n"))
                                            .unwrap_or(from_child.clone());
                                    } else if pending_tool_list_ids.remove(&id_key) {
                                        patch_tools_list_response(&mut parsed);
                                        outbound = serde_json::to_string(&parsed)
                                            .map(|s| format!("{s}\n"))
                                            .unwrap_or(from_child.clone());
                                    }
                                }
                            // Forward to client
                            proxy_stdout.write_all(outbound.as_bytes()).await.ok();
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
        let notification = proxy_tools_list_changed_notification();
        let notif_str = serde_json::to_string(&notification)
            .map(|s| format!("{s}\n"))
            .map_err(|e| format!("failed to encode tools/list_changed notification: {e}"))?;
        tokio::io::stdout()
            .write_all(notif_str.as_bytes())
            .await
            .map_err(|e| format!("failed to notify client about tool list change: {e}"))?;
        tokio::io::stdout()
            .flush()
            .await
            .map_err(|e| format!("failed to flush tool list change notification: {e}"))?;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MirTools, encode_hex, parse_hex_input, patch_initialize_response,
        patch_tools_list_response, proxy_reload_tool,
    };
    use serde_json::{Value as JsonValue, json};

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

    #[test]
    fn proxy_reload_tool_schema_is_valid() {
        let tool = proxy_reload_tool();
        assert_eq!(tool.get("name").and_then(|v| v.as_str()), Some("reload"));
        assert_eq!(
            tool.pointer("/inputSchema/type").and_then(|v| v.as_str()),
            Some("object")
        );
        assert_eq!(
            tool.pointer("/inputSchema/additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn patch_tools_list_response_appends_reload_once() {
        let mut response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": MirTools::tools()
            }
        });
        patch_tools_list_response(&mut response);
        patch_tools_list_response(&mut response);
        let tools = response["result"]["tools"].as_array().unwrap();
        let reload_count = tools
            .iter()
            .filter(|tool| tool.get("name") == Some(&JsonValue::String("reload".to_owned())))
            .count();
        assert_eq!(reload_count, 1);
    }

    #[test]
    fn patch_initialize_response_enables_list_changed() {
        let mut response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "capabilities": {
                    "tools": {
                        "listChanged": false
                    }
                }
            }
        });
        patch_initialize_response(&mut response);
        assert_eq!(
            response["result"]["capabilities"]["tools"]["listChanged"],
            JsonValue::Bool(true)
        );
    }
}
