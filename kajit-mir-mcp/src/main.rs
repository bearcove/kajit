use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
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
        SessionInspectOutputTool
    ]
);

#[derive(Default)]
struct ServerState {
    sessions: HashMap<u64, DebuggerSession>,
    next_session_id: u64,
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
            name: "kajit-mir-mcp".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            description: Some(
                "Session-based CFG-MIR debugger (step/run/inspect) for kajit-mir".into(),
            ),
            title: Some("Kajit MIR Debugger".into()),
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
            "CFG-MIR interpreter debugger over MCP. Use tools/list then tools/call.".into(),
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

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--real") {
        // Real mode: run the actual MCP server
        if let Err(error) = run().await {
            eprintln!("{error}");
            std::process::exit(1);
        }
    } else {
        // Proxy mode: spawn --real subprocess and forward stdin/stdout
        if let Err(error) = run_proxy().await {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
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
            .arg("--real")
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
        eprintln!("[kajit-mir-mcp] reloading backend...");
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
