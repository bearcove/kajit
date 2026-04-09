use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use chumsky::error::Rich;
use kajit_reprs::schema_poc;
use kajit_reprs::schema_poc::hir::{
    self as schema_hir, Block, Expr, Function, Local, Module, Param, Place, Prov, Stmt, TypeDef,
};
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

type DocumentMap = Arc<RwLock<HashMap<Url, DocumentState>>>;

#[derive(Clone)]
struct DocumentState {
    content: String,
    version: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum KajitSemanticTokenType {
    Keyword = 0,
    String = 1,
    Number = 2,
    Type = 3,
    Function = 4,
    Parameter = 5,
    Variable = 6,
    Property = 7,
    EnumMember = 8,
    Operator = 9,
}

impl KajitSemanticTokenType {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Keyword => "keyword",
            Self::String => "string",
            Self::Number => "number",
            Self::Type => "type",
            Self::Function => "function",
            Self::Parameter => "parameter",
            Self::Variable => "variable",
            Self::Property => "property",
            Self::EnumMember => "enumMember",
            Self::Operator => "operator",
        }
    }
}

#[derive(Debug)]
struct RawSemanticToken {
    line: u32,
    start_char: u32,
    length: u32,
    token_type: KajitSemanticTokenType,
}

pub async fn cmd_lsp(stdio: bool) {
    if !stdio {
        eprintln!("error: kajit lsp currently only supports --stdio");
        std::process::exit(2);
    }

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(KajitLanguageServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}

struct KajitLanguageServer {
    client: Client,
    documents: DocumentMap,
}

impl KajitLanguageServer {
    fn new(client: Client) -> Self {
        Self {
            client,
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn update_document(&self, uri: Url, content: String, version: i32) {
        let diagnostics = compute_hir_diagnostics(&content);
        self.client
            .publish_diagnostics(uri.clone(), diagnostics, Some(version))
            .await;

        let mut docs = self.documents.write().await;
        docs.insert(uri, DocumentState { content, version });
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for KajitLanguageServer {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(
                        SemanticTokensOptions {
                            work_done_progress_options: WorkDoneProgressOptions::default(),
                            legend: semantic_token_legend(),
                            range: Some(false),
                            full: Some(SemanticTokensFullOptions::Bool(true)),
                        },
                    ),
                ),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "kajit-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(
                MessageType::INFO,
                format!(
                    "Kajit language server initialized (PID: {})",
                    std::process::id()
                ),
            )
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.update_document(
            params.text_document.uri,
            params.text_document.text,
            params.text_document.version,
        )
        .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.into_iter().next() {
            self.update_document(
                params.text_document.uri,
                change.text,
                params.text_document.version,
            )
            .await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let mut docs = self.documents.write().await;
        docs.remove(&params.text_document.uri);
        self.client
            .publish_diagnostics(params.text_document.uri, Vec::new(), None)
            .await;
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        if !is_hir_uri(&params.text_document.uri) {
            return Ok(None);
        }

        let maybe_doc = {
            let docs = self.documents.read().await;
            docs.get(&params.text_document.uri).cloned()
        };

        let (content, result_id) = if let Some(doc) = maybe_doc {
            (doc.content, Some(doc.version.to_string()))
        } else if let Ok(path) = params.text_document.uri.to_file_path() {
            match fs::read_to_string(&path) {
                Ok(content) => (content, None),
                Err(_) => return Ok(None),
            }
        } else {
            return Ok(None);
        };

        let tokens = compute_hir_semantic_tokens(&content);
        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id,
            data: tokens,
        })))
    }
}

fn semantic_token_legend() -> SemanticTokensLegend {
    SemanticTokensLegend {
        token_types: vec![
            SemanticTokenType::new(KajitSemanticTokenType::Keyword.as_str()),
            SemanticTokenType::new(KajitSemanticTokenType::String.as_str()),
            SemanticTokenType::new(KajitSemanticTokenType::Number.as_str()),
            SemanticTokenType::new(KajitSemanticTokenType::Type.as_str()),
            SemanticTokenType::new(KajitSemanticTokenType::Function.as_str()),
            SemanticTokenType::new(KajitSemanticTokenType::Parameter.as_str()),
            SemanticTokenType::new(KajitSemanticTokenType::Variable.as_str()),
            SemanticTokenType::new(KajitSemanticTokenType::Property.as_str()),
            SemanticTokenType::new(KajitSemanticTokenType::EnumMember.as_str()),
            SemanticTokenType::new(KajitSemanticTokenType::Operator.as_str()),
        ],
        token_modifiers: vec![],
    }
}

fn is_hir_uri(uri: &Url) -> bool {
    uri.to_file_path().ok().is_some_and(|path| {
        schema_poc::REPRS
            .iter()
            .any(|repr| path_matches_ext(&path, repr.file_ext))
    })
}

fn path_matches_ext(path: &std::path::Path, file_ext: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(file_ext))
}

fn compute_hir_diagnostics(content: &str) -> Vec<Diagnostic> {
    match schema_hir::parse_module_text_rich(content, None) {
        Ok(_) => Vec::new(),
        Err(errors) => errors
            .into_iter()
            .map(|error: Rich<'_, char>| {
                let span = error.span();
                Diagnostic {
                    range: Range {
                        start: offset_to_position(content, span.start),
                        end: offset_to_position(content, span.end),
                    },
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: None,
                    code_description: None,
                    source: Some("kajit-schema-poc-hir".to_string()),
                    message: error.to_string(),
                    related_information: None,
                    tags: None,
                    data: None,
                }
            })
            .collect(),
    }
}

fn compute_hir_semantic_tokens(content: &str) -> Vec<SemanticToken> {
    let Ok(module) = schema_hir::parse_module_text_rich(content, None) else {
        return Vec::new();
    };

    let mut raw = Vec::new();
    if let Some(range) = find_in_text(content, "module") {
        push_range_token(&mut raw, range, KajitSemanticTokenType::Keyword);
    }
    collect_module_tokens(content, &module, &mut raw);

    raw.sort_by(|a, b| a.line.cmp(&b.line).then(a.start_char.cmp(&b.start_char)));
    raw.dedup_by(|a, b| {
        a.line == b.line
            && a.start_char == b.start_char
            && a.length == b.length
            && a.token_type == b.token_type
    });
    encode_semantic_tokens(&raw)
}

fn collect_module_tokens(content: &str, module: &Module, raw: &mut Vec<RawSemanticToken>) {
    for type_def in &module.type_defs {
        collect_type_def_tokens(content, type_def, raw);
    }
    for function in &module.functions {
        collect_function_tokens(content, function, raw);
    }
}

fn collect_type_def_tokens(content: &str, type_def: &TypeDef, raw: &mut Vec<RawSemanticToken>) {
    if let Some(range) = find_in_prov(content, &type_def.prov, &type_def.name.0) {
        push_range_token(raw, range, KajitSemanticTokenType::Type);
    }
}

fn collect_function_tokens(content: &str, function: &Function, raw: &mut Vec<RawSemanticToken>) {
    if let Some(range) = find_in_prov(content, &function.prov, "fn") {
        push_range_token(raw, range, KajitSemanticTokenType::Keyword);
    }
    if let Some(range) = find_after_in_prov(content, &function.prov, "fn", &function.name.0) {
        push_range_token(raw, range, KajitSemanticTokenType::Function);
    }
    if let Some(range) = find_after_in_prov(content, &function.prov, "->", &function.return_type.0)
    {
        push_range_token(raw, range, KajitSemanticTokenType::Type);
    }
    for param in &function.params {
        collect_param_tokens(content, param, raw);
    }
    for local in &function.locals {
        collect_local_tokens(content, local, raw);
    }
    collect_block_tokens(content, &function.body, raw);
}

fn collect_param_tokens(content: &str, param: &Param, raw: &mut Vec<RawSemanticToken>) {
    if let Some(range) = find_in_prov(content, &param.prov, &param.name.0) {
        push_range_token(raw, range, KajitSemanticTokenType::Parameter);
    }
    if let Some(range) = find_after_in_prov(content, &param.prov, ":", &param.ty.0) {
        push_range_token(raw, range, KajitSemanticTokenType::Type);
    }
}

fn collect_local_tokens(content: &str, local: &Local, raw: &mut Vec<RawSemanticToken>) {
    if let Some(range) = find_in_prov(content, &local.prov, &local.name.0) {
        push_range_token(raw, range, KajitSemanticTokenType::Variable);
    }
    if let Some(range) = find_after_in_prov(content, &local.prov, ":", &local.ty.0) {
        push_range_token(raw, range, KajitSemanticTokenType::Type);
    }
}

fn collect_block_tokens(content: &str, block: &Block, raw: &mut Vec<RawSemanticToken>) {
    for stmt in &block.statements {
        collect_stmt_tokens(content, stmt, raw);
    }
}

fn collect_stmt_tokens(content: &str, stmt: &Stmt, raw: &mut Vec<RawSemanticToken>) {
    match stmt {
        Stmt::Assign { place, value, .. } => {
            collect_place_tokens(content, place, raw);
            collect_expr_tokens(content, value, raw);
        }
        Stmt::Expr { value, .. } => collect_expr_tokens(content, value, raw),
        Stmt::If {
            condition,
            then,
            r#else,
            prov,
        } => {
            if let Some(range) = find_in_prov(content, prov, "if") {
                push_range_token(raw, range, KajitSemanticTokenType::Keyword);
            }
            if let Some(range) = find_in_prov(content, prov, "else") {
                push_range_token(raw, range, KajitSemanticTokenType::Keyword);
            }
            collect_expr_tokens(content, condition, raw);
            collect_block_tokens(content, then, raw);
            if let Some(r#else) = r#else {
                collect_block_tokens(content, r#else, raw);
            }
        }
        Stmt::Init { place, value, prov } => {
            if let Some(range) = find_in_prov(content, prov, "init") {
                push_range_token(raw, range, KajitSemanticTokenType::Keyword);
            }
            collect_place_tokens(content, place, raw);
            collect_expr_tokens(content, value, raw);
        }
        Stmt::Return { value, prov } => {
            if let Some(range) = find_in_prov(content, prov, "return") {
                push_range_token(raw, range, KajitSemanticTokenType::Keyword);
            }
            if let Some(value) = value {
                collect_expr_tokens(content, value, raw);
            }
        }
    }
}

fn collect_expr_tokens(content: &str, expr: &Expr, raw: &mut Vec<RawSemanticToken>) {
    match expr {
        Expr::Binary { lhs, rhs, prov, .. } => {
            collect_expr_tokens(content, lhs, raw);
            if let Some(range) = find_any_operator_in_prov(content, prov) {
                push_range_token(raw, range, KajitSemanticTokenType::Operator);
            }
            collect_expr_tokens(content, rhs, raw);
        }
        Expr::Call { args, callee, prov } => {
            if let Some(range) = find_in_prov(content, prov, "call") {
                push_range_token(raw, range, KajitSemanticTokenType::Keyword);
            }
            if let Some(range) = find_after_in_prov(content, prov, "call", &callee.0) {
                push_range_token(raw, range, KajitSemanticTokenType::Function);
            }
            for arg in args {
                collect_expr_tokens(content, arg, raw);
            }
        }
        Expr::Field { base, field, prov } => {
            collect_expr_tokens(content, base, raw);
            if let Some(range) = find_last_in_prov(content, prov, &field.0) {
                push_range_token(raw, range, KajitSemanticTokenType::Property);
            }
        }
        Expr::Literal { prov, .. } => {
            if let Some(range) = range_from_prov(content, prov) {
                push_range_token(raw, range, KajitSemanticTokenType::Number);
            }
        }
        Expr::Local { prov, .. } => {
            if let Some(range) = range_from_prov(content, prov) {
                push_range_token(raw, range, KajitSemanticTokenType::Variable);
            }
        }
    }
}

fn collect_place_tokens(content: &str, place: &Place, raw: &mut Vec<RawSemanticToken>) {
    match place {
        Place::Field { base, field, prov } => {
            collect_place_tokens(content, base, raw);
            if let Some(range) = find_last_in_prov(content, prov, &field.0) {
                push_range_token(raw, range, KajitSemanticTokenType::Property);
            }
        }
        Place::Local { prov, .. } => {
            if let Some(range) = range_from_prov(content, prov) {
                push_range_token(raw, range, KajitSemanticTokenType::Variable);
            }
        }
    }
}

fn range_from_prov(content: &str, prov: &Prov) -> Option<Range> {
    let span = prov.span.as_ref()?;
    Some(Range {
        start: offset_to_position(content, span.start as usize),
        end: offset_to_position(content, span.end as usize),
    })
}

fn span_bytes(prov: &Prov, content: &str) -> Option<(usize, usize)> {
    let span = prov.span.as_ref()?;
    let start = span.start as usize;
    let end = span.end as usize;
    if start > end || end > content.len() {
        return None;
    }
    Some((start, end))
}

fn find_in_text(content: &str, needle: &str) -> Option<Range> {
    let start = content.find(needle)?;
    Some(byte_range_to_range(content, start, start + needle.len()))
}

fn find_in_prov(content: &str, prov: &Prov, needle: &str) -> Option<Range> {
    find_in_span(content, span_bytes(prov, content)?, needle)
}

fn find_last_in_prov(content: &str, prov: &Prov, needle: &str) -> Option<Range> {
    find_last_in_span(content, span_bytes(prov, content)?, needle)
}

fn find_after_in_prov(content: &str, prov: &Prov, anchor: &str, needle: &str) -> Option<Range> {
    find_after_in_span(content, span_bytes(prov, content)?, anchor, needle)
}

fn find_in_span(content: &str, (start, end): (usize, usize), needle: &str) -> Option<Range> {
    let offset = content.get(start..end)?.find(needle)?;
    Some(byte_range_to_range(
        content,
        start + offset,
        start + offset + needle.len(),
    ))
}

fn find_last_in_span(content: &str, (start, end): (usize, usize), needle: &str) -> Option<Range> {
    let offset = content.get(start..end)?.rfind(needle)?;
    Some(byte_range_to_range(
        content,
        start + offset,
        start + offset + needle.len(),
    ))
}

fn find_after_in_span(
    content: &str,
    (start, end): (usize, usize),
    anchor: &str,
    needle: &str,
) -> Option<Range> {
    let slice = content.get(start..end)?;
    let anchor_start = slice.find(anchor)?;
    let search_start = start + anchor_start + anchor.len();
    let offset = content.get(search_start..end)?.find(needle)?;
    Some(byte_range_to_range(
        content,
        search_start + offset,
        search_start + offset + needle.len(),
    ))
}

fn find_any_operator_in_prov(content: &str, prov: &Prov) -> Option<Range> {
    const OPERATORS: &[&str] = &["==", "!=", "<=", ">=", "+", "-", "*", "/", "&&", "||"];
    for operator in OPERATORS {
        if let Some(range) = find_in_prov(content, prov, operator) {
            return Some(range);
        }
    }
    None
}

fn byte_range_to_range(content: &str, start: usize, end: usize) -> Range {
    Range {
        start: offset_to_position(content, start),
        end: offset_to_position(content, end),
    }
}

fn push_range_token(
    raw: &mut Vec<RawSemanticToken>,
    range: Range,
    token_type: KajitSemanticTokenType,
) {
    if range.start.line != range.end.line {
        return;
    }
    let length = range.end.character.saturating_sub(range.start.character);
    if length == 0 {
        return;
    }
    raw.push(RawSemanticToken {
        line: range.start.line,
        start_char: range.start.character,
        length,
        token_type,
    });
}

fn offset_to_position(content: &str, offset: usize) -> Position {
    let mut line = 0_u32;
    let mut col = 0_u32;
    for ch in content[..offset.min(content.len())].chars() {
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    Position::new(line, col)
}

fn encode_semantic_tokens(raw_tokens: &[RawSemanticToken]) -> Vec<SemanticToken> {
    let mut result = Vec::with_capacity(raw_tokens.len());
    let mut prev_line = 0;
    let mut prev_start = 0;

    for token in raw_tokens {
        let delta_line = token.line.saturating_sub(prev_line);
        let delta_start = if delta_line == 0 {
            token.start_char.saturating_sub(prev_start)
        } else {
            token.start_char
        };

        result.push(SemanticToken {
            delta_line,
            delta_start,
            length: token.length,
            token_type: token.token_type as u32,
            token_modifiers_bitset: 0,
        });

        prev_line = token.line;
        prev_start = token.start_char;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_hir_semantic_tokens() {
        let source = "module { fn main() -> Value { return 42 } }";
        let tokens = compute_hir_semantic_tokens(source);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn reports_parser_errors() {
        let source = "module { fn broken( -> Value { return } }";
        let diagnostics = compute_hir_diagnostics(source);
        assert!(!diagnostics.is_empty());
    }
}
