use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use chumsky::error::Rich;
use kajit_reprs::asm::{self as schema_asm, Program as AsmProgram};
use kajit_reprs::hir::{self as schema_hir, Module};
use kajit_reprs::{
    ResolutionSet, ResolvedRef, SemanticToken as GeneratedSemanticToken, SymbolDef,
    SymbolKind as ResolvedSymbolKind,
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
enum ReprKind {
    Hir,
    Asm,
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
    Label = 10,
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
            Self::Label => "label",
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
        let diagnostics = compute_diagnostics(&uri, &content);
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
                document_formatting_provider: Some(OneOf::Left(true)),
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
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
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
        let Some(repr_kind) = repr_kind_for_uri(&params.text_document.uri) else {
            return Ok(None);
        };

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

        let tokens = compute_semantic_tokens(repr_kind, &content);
        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id,
            data: tokens,
        })))
    }

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let Some(repr_kind) = repr_kind_for_uri(&params.text_document.uri) else {
            return Ok(None);
        };

        let maybe_doc = {
            let docs = self.documents.read().await;
            docs.get(&params.text_document.uri).cloned()
        };

        let content = if let Some(doc) = maybe_doc {
            doc.content
        } else if let Ok(path) = params.text_document.uri.to_file_path() {
            match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(_) => return Ok(None),
            }
        } else {
            return Ok(None);
        };

        Ok(format_document(repr_kind, &content).map(|edit| vec![edit]))
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let Some(repr_kind) = repr_kind_for_uri(uri) else {
            return Ok(None);
        };

        let maybe_doc = {
            let docs = self.documents.read().await;
            docs.get(uri).cloned()
        };

        let content = if let Some(doc) = maybe_doc {
            doc.content
        } else if let Ok(path) = uri.to_file_path() {
            match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(_) => return Ok(None),
            }
        } else {
            return Ok(None);
        };

        Ok(compute_hover(
            repr_kind,
            &content,
            params.text_document_position_params.position,
        ))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let Some(repr_kind) = repr_kind_for_uri(uri) else {
            return Ok(None);
        };

        let maybe_doc = {
            let docs = self.documents.read().await;
            docs.get(uri).cloned()
        };

        let content = if let Some(doc) = maybe_doc {
            doc.content
        } else if let Ok(path) = uri.to_file_path() {
            match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(_) => return Ok(None),
            }
        } else {
            return Ok(None);
        };

        Ok(compute_goto_definition(
            repr_kind,
            uri,
            &content,
            params.text_document_position_params.position,
        ))
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;
        let Some(repr_kind) = repr_kind_for_uri(uri) else {
            return Ok(None);
        };

        let maybe_doc = {
            let docs = self.documents.read().await;
            docs.get(uri).cloned()
        };

        let content = if let Some(doc) = maybe_doc {
            doc.content
        } else if let Ok(path) = uri.to_file_path() {
            match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(_) => return Ok(None),
            }
        } else {
            return Ok(None);
        };

        Ok(compute_document_symbols(repr_kind, &content).map(DocumentSymbolResponse::Nested))
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
            SemanticTokenType::new(KajitSemanticTokenType::Label.as_str()),
        ],
        token_modifiers: vec![],
    }
}

fn repr_kind_for_uri(uri: &Url) -> Option<ReprKind> {
    let path = uri.to_file_path().ok()?;
    if path_matches_ext(&path, schema_hir::REPR_FILE_EXT) {
        Some(ReprKind::Hir)
    } else if path_matches_ext(&path, schema_asm::REPR_FILE_EXT) {
        Some(ReprKind::Asm)
    } else {
        None
    }
}

fn path_matches_ext(path: &std::path::Path, file_ext: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(file_ext))
}

fn compute_diagnostics(uri: &Url, content: &str) -> Vec<Diagnostic> {
    match repr_kind_for_uri(uri) {
        Some(ReprKind::Hir) => compute_hir_diagnostics(content),
        Some(ReprKind::Asm) => compute_asm_diagnostics(content),
        None => Vec::new(),
    }
}

fn compute_hir_diagnostics(content: &str) -> Vec<Diagnostic> {
    match schema_hir::parse_root_text_rich(content, None) {
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

fn compute_asm_diagnostics(content: &str) -> Vec<Diagnostic> {
    match schema_asm::parse_root_text_rich(content, None) {
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
                    source: Some("kajit-schema-poc-asm".to_string()),
                    message: error.to_string(),
                    related_information: None,
                    tags: None,
                    data: None,
                }
            })
            .collect(),
    }
}

fn compute_semantic_tokens(repr_kind: ReprKind, content: &str) -> Vec<SemanticToken> {
    let generated = match repr_kind {
        ReprKind::Hir => schema_hir::semantic_tokens(content),
        ReprKind::Asm => schema_asm::semantic_tokens(content),
    };
    encode_semantic_tokens(content, &generated)
}

fn format_document(repr_kind: ReprKind, content: &str) -> Option<TextEdit> {
    let formatted = match repr_kind {
        ReprKind::Hir => {
            let module: Module = schema_hir::parse_root_text(content).ok()?;
            schema_hir::format_root_text(&module)
        }
        ReprKind::Asm => {
            let program: AsmProgram = schema_asm::parse_root_text(content).ok()?;
            schema_asm::format_root_text(&program)
        }
    };
    if formatted == content {
        return None;
    }

    Some(TextEdit {
        range: Range {
            start: Position::new(0, 0),
            end: offset_to_position(content, content.len()),
        },
        new_text: formatted,
    })
}

fn generated_kind(kind: &'static str) -> Option<KajitSemanticTokenType> {
    match kind {
        "keyword" => Some(KajitSemanticTokenType::Keyword),
        "string" => Some(KajitSemanticTokenType::String),
        "number" => Some(KajitSemanticTokenType::Number),
        "type" => Some(KajitSemanticTokenType::Type),
        "function" => Some(KajitSemanticTokenType::Function),
        "parameter" => Some(KajitSemanticTokenType::Parameter),
        "variable" => Some(KajitSemanticTokenType::Variable),
        "property" => Some(KajitSemanticTokenType::Property),
        "enumMember" => Some(KajitSemanticTokenType::EnumMember),
        "operator" => Some(KajitSemanticTokenType::Operator),
        "label" => Some(KajitSemanticTokenType::Label),
        _ => None,
    }
}

fn position_to_offset(content: &str, position: Position) -> usize {
    let target_line = position.line as usize;
    let target_col = position.character as usize;
    let mut line = 0usize;
    let mut col = 0usize;

    for (idx, ch) in content.char_indices() {
        if line == target_line && col == target_col {
            return idx;
        }
        if ch == '\n' {
            if line == target_line {
                return idx;
            }
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    content.len()
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

fn encode_semantic_tokens(
    content: &str,
    generated_tokens: &[GeneratedSemanticToken],
) -> Vec<SemanticToken> {
    let raw_tokens = generated_tokens
        .iter()
        .filter_map(|token| {
            let token_type = generated_kind(token.kind)?;
            let start = offset_to_position(content, token.start as usize);
            let end = offset_to_position(content, token.end as usize);
            if start.line != end.line {
                return None;
            }
            let length = end.character.saturating_sub(start.character);
            if length == 0 {
                return None;
            }
            Some(RawSemanticToken {
                line: start.line,
                start_char: start.character,
                length,
                token_type,
            })
        })
        .collect::<Vec<_>>();
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

fn compute_hover(repr_kind: ReprKind, content: &str, position: Position) -> Option<Hover> {
    let offset = position_to_offset(content, position) as u32;
    if let Some(hover) = compute_resolved_symbol_hover(repr_kind, content, position) {
        return Some(hover);
    }
    let generated = match repr_kind {
        ReprKind::Hir => schema_hir::hover_entries(content),
        ReprKind::Asm => schema_asm::hover_entries(content),
    };
    let best = generated
        .iter()
        .filter(|entry| entry.start <= offset && offset < entry.end)
        .min_by_key(|entry| (entry.end - entry.start, std::cmp::Reverse(entry.priority)))?;

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: best.markdown.clone(),
        }),
        range: Some(Range {
            start: offset_to_position(content, best.start as usize),
            end: offset_to_position(content, best.end as usize),
        }),
    })
}

fn compute_resolved_symbol_hover(
    repr_kind: ReprKind,
    content: &str,
    position: Position,
) -> Option<Hover> {
    let offset = position_to_offset(content, position) as u32;
    let resolutions = resolve_document(repr_kind, content)?;
    let (reference, target) = find_resolved_reference(&resolutions, offset)?;
    let definition = resolutions.definitions.get(target?)?;
    let markdown = render_symbol_hover(definition)?;
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: markdown,
        }),
        range: Some(Range {
            start: offset_to_position(content, reference.reference.start as usize),
            end: offset_to_position(content, reference.reference.end as usize),
        }),
    })
}

fn compute_goto_definition(
    repr_kind: ReprKind,
    uri: &Url,
    content: &str,
    position: Position,
) -> Option<GotoDefinitionResponse> {
    let offset = position_to_offset(content, position) as u32;
    let resolutions = resolve_document(repr_kind, content)?;
    let (_, target) = find_resolved_reference(&resolutions, offset)?;
    let definition = resolutions.definitions.get(target?)?;
    Some(GotoDefinitionResponse::Scalar(Location {
        uri: uri.clone(),
        range: Range {
            start: offset_to_position(content, definition.start as usize),
            end: offset_to_position(content, definition.end as usize),
        },
    }))
}

fn resolve_document(repr_kind: ReprKind, content: &str) -> Option<ResolutionSet> {
    match repr_kind {
        ReprKind::Hir => schema_hir::resolve(content).ok(),
        ReprKind::Asm => schema_asm::resolve(content).ok(),
    }
}

fn find_resolved_reference(
    resolutions: &ResolutionSet,
    offset: u32,
) -> Option<(&ResolvedRef, Option<usize>)> {
    resolutions
        .references
        .iter()
        .filter(|reference| reference.reference.start <= offset && offset < reference.reference.end)
        .min_by_key(|reference| reference.reference.end - reference.reference.start)
        .map(|reference| (reference, reference.target))
}

fn render_symbol_hover(definition: &SymbolDef) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(detail) = &definition.detail {
        parts.push(format!("```kajit\n{detail}\n```"));
    } else {
        parts.push(format!("`{}`", definition.name));
    }
    if let Some(docs) = &definition.docs
        && !docs.trim().is_empty()
    {
        parts.push(docs.clone());
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

fn compute_document_symbols(repr_kind: ReprKind, content: &str) -> Option<Vec<DocumentSymbol>> {
    let resolutions = resolve_document(repr_kind, content)?;
    let mut symbols = resolutions
        .definitions
        .iter()
        .map(|definition| DocumentSymbol {
            name: definition.name.clone(),
            detail: definition.detail.clone(),
            kind: symbol_kind_to_document_symbol_kind(definition.kind),
            tags: None,
            deprecated: None,
            range: Range {
                start: offset_to_position(content, definition.start as usize),
                end: offset_to_position(content, definition.end as usize),
            },
            selection_range: Range {
                start: offset_to_position(content, definition.start as usize),
                end: offset_to_position(content, definition.end as usize),
            },
            children: None,
        })
        .collect::<Vec<_>>();
    symbols.sort_by_key(|symbol| {
        (
            symbol.range.start.line,
            symbol.range.start.character,
            symbol.range.end.line,
            symbol.range.end.character,
        )
    });
    Some(symbols)
}

fn symbol_kind_to_document_symbol_kind(kind: ResolvedSymbolKind) -> SymbolKind {
    match kind {
        ResolvedSymbolKind::Function => SymbolKind::FUNCTION,
        ResolvedSymbolKind::Type => SymbolKind::CLASS,
        ResolvedSymbolKind::Label => SymbolKind::KEY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_hir_semantic_tokens() {
        let source = "module { fn main() -> Value { return 42 } }";
        let tokens = compute_semantic_tokens(ReprKind::Hir, source);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn produces_asm_semantic_tokens() {
        let source = "asm aarch64 { entry: movz x0, 42 ret }";
        let tokens = compute_semantic_tokens(ReprKind::Asm, source);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn reports_parser_errors() {
        let source = "module { fn broken( -> Value { return } }";
        let diagnostics = compute_hir_diagnostics(source);
        assert!(!diagnostics.is_empty());
    }

    #[test]
    fn formats_hir_document() {
        let source = "module { fn main() -> Value { return 42 } }";
        let edit = format_document(ReprKind::Hir, source).expect("expected formatting edit");
        assert_eq!(edit.range.start, Position::new(0, 0));
        assert_eq!(edit.range.end, offset_to_position(source, source.len()));
        assert_eq!(
            edit.new_text,
            "module {\n    fn main() -> Value {\n        return 42\n    }\n}"
        );
    }

    #[test]
    fn reports_asm_parser_errors() {
        let source = "asm aarch64 { entry: movz x0, ret }";
        let diagnostics = compute_asm_diagnostics(source);
        assert!(!diagnostics.is_empty());
    }

    #[test]
    fn formats_asm_document() {
        let source = "asm aarch64 { entry: movz x0, 42 ret }";
        let edit = format_document(ReprKind::Asm, source).expect("expected formatting edit");
        assert_eq!(edit.range.start, Position::new(0, 0));
        assert_eq!(edit.range.end, offset_to_position(source, source.len()));
        assert_eq!(
            edit.new_text,
            "asm aarch64 {\n    entry:\n    movz x0, 42\n    ret\n}"
        );
    }

    #[test]
    fn hovers_asm_keyword_from_schema_docs() {
        let source = "asm x86_64 {\n    entry:\n    mov rax, 42\n    ret\n}";
        let hover = compute_hover(ReprKind::Asm, source, Position::new(2, 4))
            .expect("expected hover on mov");
        let HoverContents::Markup(markup) = hover.contents else {
            panic!("expected markdown hover");
        };
        assert!(markup.value.contains("Move an immediate into a register."));
    }

    #[test]
    fn hovers_asm_register_from_schema_docs() {
        let source = "asm x86_64 {\n    entry:\n    mov rax, 42\n    ret\n}";
        let hover = compute_hover(ReprKind::Asm, source, Position::new(2, 8))
            .expect("expected hover on rax");
        let HoverContents::Markup(markup) = hover.contents else {
            panic!("expected markdown hover");
        };
        assert!(
            markup
                .value
                .contains("Return-value and accumulator register.")
        );
    }

    #[test]
    fn hovers_hir_function_reference_from_resolver() {
        let source = "module {\n    fn build_vec(value: Value) -> Value {\n        return value\n    }\n\n    fn main(value: Value) -> Value {\n        return call @build_vec(value)\n    }\n}";
        let hover = compute_hover(ReprKind::Hir, source, Position::new(6, 22))
            .expect("expected hover on build_vec call");
        let HoverContents::Markup(markup) = hover.contents else {
            panic!("expected markdown hover");
        };
        assert!(markup.value.contains("fn build_vec(value: Value) -> Value"));
    }

    #[test]
    fn goto_definition_for_asm_label_uses_shared_resolver() {
        let source = "asm x86_64 {\n    entry:\n    jmp entry\n}";
        let uri = Url::parse("file:///tmp/test.k-asm").expect("valid uri");
        let response = compute_goto_definition(ReprKind::Asm, &uri, source, Position::new(2, 8))
            .expect("expected label definition");
        let GotoDefinitionResponse::Scalar(location) = response else {
            panic!("expected scalar location");
        };
        assert_eq!(location.uri, uri);
        assert_eq!(location.range.start, Position::new(1, 4));
        assert_eq!(location.range.end, Position::new(1, 9));
    }

    #[test]
    fn goto_definition_for_hir_function_uses_shared_resolver() {
        let source = "module {\n    fn build_vec(value: Value) -> Value {\n        return value\n    }\n\n    fn main(value: Value) -> Value {\n        return call @build_vec(value)\n    }\n}";
        let uri = Url::parse("file:///tmp/test.k-hir").expect("valid uri");
        let response = compute_goto_definition(ReprKind::Hir, &uri, source, Position::new(6, 22))
            .expect("expected function definition");
        let GotoDefinitionResponse::Scalar(location) = response else {
            panic!("expected scalar location");
        };
        assert_eq!(location.uri, uri);
        assert_eq!(location.range.start, Position::new(1, 7));
        assert_eq!(location.range.end, Position::new(1, 16));
    }

    #[test]
    fn document_symbols_come_from_shared_resolver_definitions() {
        let source =
            "module {\n    fn build_vec(value: Value) -> Value {\n        return value\n    }\n}";
        let symbols = compute_document_symbols(ReprKind::Hir, source).expect("expected symbols");
        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name, "build_vec");
        assert_eq!(symbols[0].kind, SymbolKind::FUNCTION);
        assert_eq!(
            symbols[0].detail.as_deref(),
            Some("fn build_vec(value: Value) -> Value")
        );
    }
}
