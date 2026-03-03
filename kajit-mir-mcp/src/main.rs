use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use kajit_mir::{BlockId, DebuggerSession, DebuggerState, RaProgram, RunUntilTarget, StepEvent};
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
    description = "Create a new debugger session from RA-MIR or IR text."
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct SessionNewTool {
    /// Program text kind: `ra_mir` or `ir`.
    input_kind: String,
    /// Program text payload matching `input_kind`.
    ///
    /// Mutually exclusive with `program_path`.
    #[serde(default)]
    program_text: Option<String>,
    /// Path to a file containing program text matching `input_kind`.
    ///
    /// Mutually exclusive with `program_text`.
    #[serde(default)]
    program_path: Option<String>,
    /// Input bytes as hex string (e.g. '8101' or '[0x81, 0x01]')
    #[serde(default)]
    input_hex: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputKind {
    RaMir,
    Ir,
}

impl InputKind {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "ra_mir" | "ra-mir" | "ramir" => Ok(Self::RaMir),
            "ir" => Ok(Self::Ir),
            other => Err(format!(
                "invalid input_kind {other:?} (expected \"ra_mir\" or \"ir\")"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::RaMir => "ra_mir",
            Self::Ir => "ir",
        }
    }
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
        let start = std::time::Instant::now();
        let input_kind = InputKind::parse(&arg_str(args, "input_kind")?)?;
        let program_text = load_program_text(args)?;
        let input_hex = arg_opt_str(args, "input_hex").unwrap_or_default();
        let input = parse_hex_input(&input_hex)?;
        let parse_start = std::time::Instant::now();
        let program = match input_kind {
            InputKind::RaMir => parse_ra_mir_text(&program_text)?,
            InputKind::Ir => lower_ir_text_to_ra_program(&program_text)?,
        };
        let parse_ms = parse_start.elapsed().as_millis() as u64;
        let session_start = std::time::Instant::now();
        let session = DebuggerSession::new(&program, &input).map_err(|e| e.to_string())?;
        let session_ms = session_start.elapsed().as_millis() as u64;

        let lock_start = std::time::Instant::now();
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
        let lock_ms = lock_start.elapsed().as_millis() as u64;
        let total_ms = start.elapsed().as_millis() as u64;
        Ok(json!({
            "session_id": session_id,
            "input_kind": input_kind.as_str(),
            "input_hex": encode_hex(&input),
            "state": state_json(&snapshot),
            "timings_ms": {
                "parse_and_lower": parse_ms,
                "session_init": session_ms,
                "state_lock_and_insert": lock_ms,
                "total": total_ms
            }
        }))
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
            .map_err(|e| e.to_string())?
            .iter()
            .map(event_json)
            .collect::<Vec<_>>();

        Ok(json!({
            "session_id": session_id,
            "events": events,
            "state": state_json(&session.state()),
            "max_steps": max_steps,
        }))
    }

    fn session_state(&self, args: &JsonMap<String, JsonValue>) -> Result<JsonValue, String> {
        let session_id = arg_u64(args, "session_id")?;
        let state = self
            .lock_state()?
            .sessions
            .get(&session_id)
            .ok_or_else(|| format!("unknown session_id: {session_id}"))?
            .state();
        Ok(json!({
            "session_id": session_id,
            "state": state_json(&state),
        }))
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
        Ok(json!({
            "session_id": session_id,
            "vreg": vreg,
            "value": value,
            "value_hex": format!("0x{value:x}"),
        }))
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
        Ok(json!({
            "session_id": session_id,
            "start": start,
            "len": bytes.len(),
            "bytes": bytes,
            "bytes_hex": encode_hex(&bytes),
        }))
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

fn load_program_text(args: &JsonMap<String, JsonValue>) -> Result<String, String> {
    let from_text = arg_opt_str(args, "program_text");
    let from_path = arg_opt_str(args, "program_path");
    match (from_text, from_path) {
        (Some(text), None) => Ok(text),
        (None, Some(path)) => std::fs::read_to_string(&path)
            .map_err(|e| format!("failed to read program_path {path:?}: {e}")),
        (Some(_), Some(_)) => {
            Err("program_text and program_path are mutually exclusive".to_owned())
        }
        (None, None) => {
            Err("missing program input: provide program_text or program_path".to_owned())
        }
    }
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
    let trap = state
        .trap
        .map(|trap| trap_json(&trap))
        .unwrap_or(JsonValue::Null);
    json!({
        "step_count": state.step_count,
        "location": {
            "block": state.location.block.0,
            "next_inst_index": state.location.next_inst_index,
            "at_terminator": state.location.at_terminator,
        },
        "cursor": state.cursor,
        "trap": trap,
        "returned": state.returned,
        "halted": state.halted,
        "vregs": state.vregs,
        "output_len": state.output.len(),
        "output_hex": encode_hex(&state.output),
    })
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
    let text = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| payload.to_string());
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
    if cleaned.len() % 2 != 0 {
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

fn parse_ra_mir_text(program_text: &str) -> Result<RaProgram, String> {
    kajit_mir_text::parse_ra_mir(program_text).map_err(|e| e.to_string())
}

fn lower_ir_text_to_ra_program(program_text: &str) -> Result<RaProgram, String> {
    if program_text.contains("0x<ptr>") {
        return Err("IR contains scrubbed intrinsic pointers (`0x<ptr>`). \
Provide named intrinsics (for example `@kajit_read_u8`) so the MCP server can resolve them."
            .to_owned());
    }
    let shape = <u8 as facet::Facet>::SHAPE;
    let registry = known_intrinsic_registry();
    let mut ir =
        kajit_ir_text::parse_ir(program_text, shape, &registry).map_err(|e| e.to_string())?;
    let linear = kajit_lir::linearize(&mut ir);
    Ok(kajit_mir::lower_linear_ir(&linear))
}

fn known_intrinsic_registry() -> kajit_ir::IntrinsicRegistry {
    let mut registry = kajit_ir::IntrinsicRegistry::empty();
    for (name, func) in kajit::intrinsics::known_intrinsics()
        .into_iter()
        .chain(kajit::json_intrinsics::known_intrinsics().into_iter())
    {
        registry.register(name, func);
    }
    registry
}

async fn run() -> Result<(), String> {
    let handler = MirHandler::default();
    let server_details = InitializeResult {
        server_info: Implementation {
            name: "kajit-mir-mcp".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            description: Some(
                "Session-based RA-MIR debugger (step/run/inspect) for kajit-mir".into(),
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
            "RA-MIR interpreter debugger over MCP. Use tools/list then tools/call.".into(),
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
    if let Err(error) = run().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        InputKind, MirTools, encode_hex, load_program_text, lower_ir_text_to_ra_program,
        parse_hex_input,
    };
    use kajit_lir::LinearOp;
    use serde_json::{Map as JsonMap, Value as JsonValue, json};

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
    fn parse_input_kind_supports_expected_aliases() {
        assert_eq!(InputKind::parse("ra_mir").unwrap(), InputKind::RaMir);
        assert_eq!(InputKind::parse("ra-mir").unwrap(), InputKind::RaMir);
        assert_eq!(InputKind::parse("ramir").unwrap(), InputKind::RaMir);
        assert_eq!(InputKind::parse("ir").unwrap(), InputKind::Ir);
    }

    #[test]
    fn parse_input_kind_rejects_unknown_values() {
        assert!(InputKind::parse("foo").is_err());
    }

    #[test]
    fn lower_ir_rejects_scrubbed_intrinsic_ptrs() {
        let input = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = CallIntrinsic(0x<ptr>, field_offset=0) [%cs:arg, %os:arg] -> [%cs, %os]
    results: [%cs:n0, %os:n0]
  }
}
"#;
        let err = lower_ir_text_to_ra_program(input).expect_err("scrubbed ptrs should fail");
        assert!(err.contains("0x<ptr>"));
    }

    #[test]
    fn lower_ir_text_to_ra_program_supports_basic_ir_input() {
        let ir_text = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = BoundsCheck(1) [%cs:arg] -> [%cs]
    n1 = ReadBytes(1) [%cs:n0] -> [v0, %cs]
    n2 = WriteToField(offset=0, W1) [v0, %os:arg] -> [%os]
    results: [%cs:n1, %os:n2]
  }
}
"#;
        let program = lower_ir_text_to_ra_program(ir_text).expect("IR should lower to RA-MIR");
        assert_eq!(program.funcs.len(), 1);
        assert!(
            !program.funcs[0].blocks.is_empty(),
            "lowered RA-MIR should contain at least one block"
        );
    }

    #[test]
    fn lower_ir_resolves_named_intrinsic_refs() {
        let ir_text = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = CallIntrinsic(@kajit_read_u8, field_offset=0) [%cs:arg, %os:arg] -> [%cs, %os]
    results: [%cs:n0, %os:n0]
  }
}
"#;

        let program =
            lower_ir_text_to_ra_program(ir_text).expect("named intrinsic refs should resolve");
        let has_nonzero_intrinsic = program.funcs.iter().any(|func| {
            func.blocks
                .iter()
                .flat_map(|block| block.insts.iter())
                .any(|inst| {
                    matches!(
                        &inst.op,
                        LinearOp::CallIntrinsic { func, .. } if func.0 != 0
                    )
                })
        });
        assert!(has_nonzero_intrinsic);
    }

    #[test]
    fn load_program_text_accepts_program_text() {
        let mut args = JsonMap::<String, JsonValue>::new();
        args.insert("program_text".to_owned(), json!("abc"));
        assert_eq!(load_program_text(&args).unwrap(), "abc");
    }

    #[test]
    fn load_program_text_rejects_both_text_and_path() {
        let mut args = JsonMap::<String, JsonValue>::new();
        args.insert("program_text".to_owned(), json!("abc"));
        args.insert("program_path".to_owned(), json!("/tmp/x"));
        assert!(load_program_text(&args).is_err());
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
