use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::str::FromStr;

use kajit_mir::{BlockId, DebuggerSession, DebuggerState, RunUntilTarget, StepEvent};
use rust_mcp_schema::schema_utils::{
    ClientJsonrpcRequest, ClientMessage, ResultFromServer, ServerJsonrpcResponse, ServerMessage,
};
use rust_mcp_schema::{
    CallToolRequest, CallToolResult, ContentBlock, Implementation, InitializeResult,
    JsonrpcErrorResponse, ListToolsResult, ProtocolVersion, RequestId, Result as McpResult,
    RpcError, ServerCapabilities, ServerCapabilitiesTools, TextContent, Tool, ToolAnnotations,
    ToolInputSchema,
};
use serde_json::{Map, Value, json};

#[derive(Default)]
struct Server {
    sessions: HashMap<u64, DebuggerSession>,
    next_session_id: u64,
}

impl Server {
    fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            next_session_id: 1,
        }
    }

    fn handle_request(&mut self, request: ClientJsonrpcRequest) -> ServerMessage {
        match request {
            ClientJsonrpcRequest::InitializeRequest(request) => {
                let result = InitializeResult {
                    capabilities: ServerCapabilities {
                        tools: Some(ServerCapabilitiesTools {
                            list_changed: Some(false),
                        }),
                        ..ServerCapabilities::default()
                    },
                    instructions: Some(
                        "RA-MIR interpreter debugger over MCP. Use tools/list then tools/call."
                            .to_owned(),
                    ),
                    meta: None,
                    protocol_version: ProtocolVersion::V2025_11_25.into(),
                    server_info: Implementation {
                        description: Some(
                            "Session-based RA-MIR debugger (step/run/inspect) for kajit-mir"
                                .to_owned(),
                        ),
                        icons: Vec::new(),
                        name: "kajit-mir-mcp".to_owned(),
                        title: Some("Kajit MIR Debugger".to_owned()),
                        version: env!("CARGO_PKG_VERSION").to_owned(),
                        website_url: None,
                    },
                };
                server_result(request.id, ResultFromServer::InitializeResult(result))
            }
            ClientJsonrpcRequest::PingRequest(request) => {
                server_result(request.id, ResultFromServer::Result(McpResult::default()))
            }
            ClientJsonrpcRequest::ListToolsRequest(request) => {
                let result = ListToolsResult {
                    meta: None,
                    next_cursor: None,
                    tools: self.tools(),
                };
                server_result(request.id, ResultFromServer::ListToolsResult(result))
            }
            ClientJsonrpcRequest::CallToolRequest(request) => self.handle_call_tool(request),
            other => server_error(
                Some(other.request_id().clone()),
                RpcError::method_not_found().with_message(format!(
                    "unsupported request method for kajit-mir-mcp: {}",
                    other.method()
                )),
            ),
        }
    }

    fn handle_call_tool(&mut self, request: CallToolRequest) -> ServerMessage {
        let name = request.params.name;
        let args = request
            .params
            .arguments
            .map(Value::Object)
            .unwrap_or_else(|| Value::Object(Map::new()));

        let result = match self.call_tool(&name, &args) {
            Ok(payload) => call_tool_ok(payload),
            Err(message) => call_tool_err(&message),
        };

        server_result(request.id, ResultFromServer::CallToolResult(result))
    }

    fn tools(&self) -> Vec<Tool> {
        vec![
            tool(
                "session_new",
                "Create a new RA-MIR debugger session.",
                &["ra_mir_text"],
                vec![
                    (
                        "ra_mir_text",
                        schema_object(json!({
                            "type": "string",
                            "description": "RA-MIR text program"
                        })),
                    ),
                    (
                        "input_hex",
                        schema_object(json!({
                            "type": "string",
                            "description": "Input bytes as hex string (e.g. '8101' or '[0x81, 0x01]')"
                        })),
                    ),
                ],
            ),
            tool(
                "session_close",
                "Close and remove a debugger session.",
                &["session_id"],
                vec![(
                    "session_id",
                    schema_object(json!({
                        "type": "integer",
                        "minimum": 1
                    })),
                )],
            ),
            tool(
                "session_step",
                "Step a session forward by one or more operations.",
                &["session_id"],
                vec![
                    (
                        "session_id",
                        schema_object(json!({"type": "integer", "minimum": 1})),
                    ),
                    (
                        "count",
                        schema_object(json!({"type": "integer", "minimum": 1})),
                    ),
                ],
            ),
            tool(
                "session_back",
                "Step a session backward by one or more recorded steps.",
                &["session_id"],
                vec![
                    (
                        "session_id",
                        schema_object(json!({"type": "integer", "minimum": 1})),
                    ),
                    (
                        "count",
                        schema_object(json!({"type": "integer", "minimum": 1})),
                    ),
                ],
            ),
            tool(
                "session_run_until",
                "Run forward until block/trap/return or max step budget.",
                &["session_id"],
                vec![
                    (
                        "session_id",
                        schema_object(json!({"type": "integer", "minimum": 1})),
                    ),
                    (
                        "block_id",
                        schema_object(json!({"type": "integer", "minimum": 0})),
                    ),
                    ("trap", schema_object(json!({"type": "boolean"}))),
                    ("return", schema_object(json!({"type": "boolean"}))),
                    (
                        "max_steps",
                        schema_object(json!({"type": "integer", "minimum": 1})),
                    ),
                ],
            ),
            tool(
                "session_state",
                "Get deterministic full state snapshot for one session.",
                &["session_id"],
                vec![(
                    "session_id",
                    schema_object(json!({"type": "integer", "minimum": 1})),
                )],
            ),
            tool(
                "session_inspect_vreg",
                "Read one virtual register by index.",
                &["session_id", "vreg"],
                vec![
                    (
                        "session_id",
                        schema_object(json!({"type": "integer", "minimum": 1})),
                    ),
                    (
                        "vreg",
                        schema_object(json!({"type": "integer", "minimum": 0})),
                    ),
                ],
            ),
            tool(
                "session_inspect_output",
                "Read a byte range from output memory.",
                &["session_id"],
                vec![
                    (
                        "session_id",
                        schema_object(json!({"type": "integer", "minimum": 1})),
                    ),
                    (
                        "start",
                        schema_object(json!({"type": "integer", "minimum": 0})),
                    ),
                    (
                        "len",
                        schema_object(json!({"type": "integer", "minimum": 0})),
                    ),
                ],
            ),
        ]
    }

    fn call_tool(&mut self, name: &str, args: &Value) -> Result<Value, String> {
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

    fn session_new(&mut self, args: &Value) -> Result<Value, String> {
        let mir_text = arg_str(args, "ra_mir_text")?;
        let input_hex = arg_opt_str(args, "input_hex").unwrap_or_default();
        let input = parse_hex_input(&input_hex)?;
        let program = kajit_mir_text::parse_ra_mir(&mir_text).map_err(|e| e.to_string())?;
        let session = DebuggerSession::new(&program, &input).map_err(|e| e.to_string())?;
        let session_id = self.next_session_id;
        self.next_session_id += 1;
        self.sessions.insert(session_id, session);

        let state = self
            .sessions
            .get(&session_id)
            .expect("inserted session should exist")
            .state();
        Ok(json!({
            "session_id": session_id,
            "input_hex": encode_hex(&input),
            "state": state_json(&state),
        }))
    }

    fn session_close(&mut self, args: &Value) -> Result<Value, String> {
        let session_id = arg_u64(args, "session_id")?;
        let removed = self.sessions.remove(&session_id).is_some();
        Ok(json!({
            "session_id": session_id,
            "closed": removed,
        }))
    }

    fn session_step(&mut self, args: &Value) -> Result<Value, String> {
        let session_id = arg_u64(args, "session_id")?;
        let count = arg_opt_u64(args, "count").unwrap_or(1) as usize;
        let session = self
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

    fn session_back(&mut self, args: &Value) -> Result<Value, String> {
        let session_id = arg_u64(args, "session_id")?;
        let count = arg_opt_u64(args, "count").unwrap_or(1) as usize;
        let session = self
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

    fn session_run_until(&mut self, args: &Value) -> Result<Value, String> {
        let session_id = arg_u64(args, "session_id")?;
        let block_id = arg_opt_u64(args, "block_id");
        let want_trap = arg_opt_bool(args, "trap").unwrap_or(false);
        let want_return = arg_opt_bool(args, "return").unwrap_or(false);
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

        let session = self
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

    fn session_state(&mut self, args: &Value) -> Result<Value, String> {
        let session_id = arg_u64(args, "session_id")?;
        let state = self
            .sessions
            .get(&session_id)
            .ok_or_else(|| format!("unknown session_id: {session_id}"))?
            .state();
        Ok(json!({
            "session_id": session_id,
            "state": state_json(&state),
        }))
    }

    fn session_inspect_vreg(&mut self, args: &Value) -> Result<Value, String> {
        let session_id = arg_u64(args, "session_id")?;
        let vreg = arg_u64(args, "vreg")? as usize;
        let session = self
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

    fn session_inspect_output(&mut self, args: &Value) -> Result<Value, String> {
        let session_id = arg_u64(args, "session_id")?;
        let start = arg_opt_u64(args, "start").unwrap_or(0) as usize;
        let len = arg_opt_u64(args, "len").unwrap_or(64) as usize;
        let session = self
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

fn schema_object(value: Value) -> Map<String, Value> {
    value
        .as_object()
        .cloned()
        .expect("schema definitions must be JSON objects")
}

fn tool(
    name: &str,
    description: &str,
    required: &[&str],
    properties: Vec<(&str, Map<String, Value>)>,
) -> Tool {
    let properties = properties
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect::<HashMap<_, _>>();
    Tool {
        annotations: Some(ToolAnnotations {
            title: Some(name.replace('_', " ")),
            read_only_hint: Some(false),
            ..ToolAnnotations::default()
        }),
        description: Some(description.to_owned()),
        execution: None,
        icons: Vec::new(),
        input_schema: ToolInputSchema::new(
            required.iter().map(|s| (*s).to_owned()).collect(),
            Some(properties),
            None,
        ),
        meta: None,
        name: name.to_owned(),
        output_schema: None,
        title: None,
    }
}

fn arg_str(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("missing or invalid string argument `{key}`"))
}

fn arg_opt_str(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(ToOwned::to_owned)
}

fn arg_u64(args: &Value, key: &str) -> Result<u64, String> {
    args.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or invalid integer argument `{key}`"))
}

fn arg_opt_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key).and_then(Value::as_u64)
}

fn arg_opt_bool(args: &Value, key: &str) -> Option<bool> {
    args.get(key).and_then(Value::as_bool)
}

fn trap_json(trap: &kajit_mir::InterpreterTrap) -> Value {
    json!({
        "code": trap.code.to_string(),
        "code_num": trap.code as u32,
        "offset": trap.offset,
    })
}

fn state_json(state: &DebuggerState) -> Value {
    let trap = state
        .trap
        .map(|trap| trap_json(&trap))
        .unwrap_or(Value::Null);
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

fn event_json(event: &StepEvent) -> Value {
    let trap = event
        .trap
        .map(|trap| trap_json(&trap))
        .unwrap_or(Value::Null);
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

fn value_to_object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => {
            let mut map = Map::new();
            map.insert("value".to_owned(), other);
            map
        }
    }
}

fn call_tool_ok(payload: Value) -> CallToolResult {
    let text = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| payload.to_string());
    CallToolResult {
        content: vec![ContentBlock::from(TextContent::new(text, None, None))],
        is_error: Some(false),
        meta: None,
        structured_content: Some(value_to_object(payload)),
    }
}

fn call_tool_err(message: &str) -> CallToolResult {
    let mut structured = Map::new();
    structured.insert("error".to_owned(), Value::String(message.to_owned()));
    CallToolResult {
        content: vec![ContentBlock::from(TextContent::new(
            message.to_owned(),
            None,
            None,
        ))],
        is_error: Some(true),
        meta: None,
        structured_content: Some(structured),
    }
}

fn server_result(id: RequestId, result: ResultFromServer) -> ServerMessage {
    ServerMessage::Response(ServerJsonrpcResponse::new(id, result))
}

fn server_error(id: Option<RequestId>, error: RpcError) -> ServerMessage {
    ServerMessage::Error(JsonrpcErrorResponse::new(error, id))
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

fn read_message<R: BufRead>(reader: &mut R) -> io::Result<Option<String>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            if content_length.is_none() {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading MCP headers",
            ));
        }

        if line == "\r\n" {
            break;
        }

        let line = line.trim_end_matches(['\r', '\n']);
        if let Some(rest) = line.strip_prefix("Content-Length:") {
            let len = rest
                .trim()
                .parse::<usize>()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            content_length = Some(len);
        }
    }

    let len = content_length.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "missing Content-Length header in MCP message",
        )
    })?;
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    String::from_utf8(body)
        .map(Some)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

fn write_message<W: Write>(writer: &mut W, body: &ServerMessage) -> io::Result<()> {
    let serialized = serde_json::to_vec(body)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    write!(writer, "Content-Length: {}\r\n\r\n", serialized.len())?;
    writer.write_all(&serialized)?;
    writer.flush()
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();
    let mut server = Server::new();

    while let Some(message) = read_message(&mut reader)? {
        let incoming = match ClientMessage::from_str(&message) {
            Ok(message) => message,
            Err(error) => {
                let response = server_error(None, error);
                write_message(&mut writer, &response)?;
                continue;
            }
        };

        let response = match incoming {
            ClientMessage::Request(request) => Some(server.handle_request(request)),
            ClientMessage::Notification(_) => None,
            ClientMessage::Response(_) | ClientMessage::Error(_) => None,
        };

        if let Some(response) = response {
            write_message(&mut writer, &response)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{encode_hex, parse_hex_input};

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
}
