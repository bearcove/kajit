use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use chumsky::Parser;
use chumsky::error::Rich;
use kajit_reprs::hir::lexer::{Span as HirSpan, Token as HirToken, lexer as hir_lexer};
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
    uri.path().ends_with(".k-hir")
}

fn compute_hir_diagnostics(content: &str) -> Vec<Diagnostic> {
    let (_, errors) = hir_lexer().parse(content).into_output_errors();

    errors
        .into_iter()
        .map(|error: Rich<'_, char, HirSpan>| {
            let span = *error.span();
            Diagnostic {
                range: span_to_range(content, span),
                severity: Some(DiagnosticSeverity::ERROR),
                code: None,
                code_description: None,
                source: Some("kajit-hir".to_string()),
                message: error.to_string(),
                related_information: None,
                tags: None,
                data: None,
            }
        })
        .collect()
}

fn compute_hir_semantic_tokens(content: &str) -> Vec<SemanticToken> {
    let (tokens, _errors) = hir_lexer().parse(content).into_output_errors();
    let Some(tokens) = tokens else {
        return Vec::new();
    };

    let mut raw = Vec::new();
    for (token, span) in tokens {
        let Some(token_type) = classify_hir_token(&token) else {
            continue;
        };

        let range = span_to_range(content, span);
        if range.start.line == range.end.line {
            let length = range.end.character.saturating_sub(range.start.character);
            if length > 0 {
                raw.push(RawSemanticToken {
                    line: range.start.line,
                    start_char: range.start.character,
                    length,
                    token_type,
                });
            }
            continue;
        }

        let start_offset = span.start;
        let end_offset = span.end;
        let mut line_start = start_offset;
        for segment in content[start_offset..end_offset].split_inclusive('\n') {
            let segment_len = segment.len();
            let segment_end = line_start + segment_len;
            let token_end = if segment.ends_with('\n') {
                segment_end.saturating_sub(1)
            } else {
                segment_end
            };
            if token_end > line_start {
                let start = offset_to_position(content, line_start);
                let end = offset_to_position(content, token_end);
                raw.push(RawSemanticToken {
                    line: start.line,
                    start_char: start.character,
                    length: end.character.saturating_sub(start.character),
                    token_type,
                });
            }
            line_start = segment_end;
        }
    }

    raw.sort_by(|a, b| a.line.cmp(&b.line).then(a.start_char.cmp(&b.start_char)));
    encode_semantic_tokens(&raw)
}

fn classify_hir_token(token: &HirToken<'_>) -> Option<KajitSemanticTokenType> {
    use HirToken::*;

    match token {
        Str(_) => Some(KajitSemanticTokenType::String),
        Int(_) | HexInt(_) => Some(KajitSemanticTokenType::Number),
        ExternSymbol(_) => Some(KajitSemanticTokenType::Function),
        LocalId(_) => Some(KajitSemanticTokenType::Variable),
        FunctionId(_) => Some(KajitSemanticTokenType::Function),
        TypeDefId(_) | KwType | KwStruct | KwEnum | KwUnit | KwBool | KwAddr | KwSlice
        | KwArray | KwStr | KwHandle => Some(KajitSemanticTokenType::Type),
        KwFunction | KwCall | KwLoad | KwDeref | KwSliceData | KwSliceLen | KwField | KwIndex
        | KwAddrOf | KwVariant | KwUnary | KwBinary | KwIf | KwElse | KwLoop | KwMatch | KwArm
        | KwBreak | KwContinue | KwReturn | KwInit | KwAssign | KwStore | KwExpr | KwParam
        | KwLet | KwTemp | KwDestination | KwParent | KwComment | KwDocs | KwScope | KwScopes
        | KwLocals | KwParams | KwCapabilities | KwControl | KwDomains | KwEffect | KwSafety
        | KwIntrinsic | KwBody | KwPure | KwReads | KwMutates | KwBarrier | KwRead | KwMutate
        | KwBuiltin | KwHost | KwReturns | KwMayFail | KwNeverReturns | KwSafeCore
        | KwOpaqueHost | KwUnsafeInterop | KwMaxIterations | KwSize | KwDiscWidth | KwW1 | KwW2
        | KwW4 | KwW8 | KwRegion | KwGenericStore | KwHirModule | KwRegions | KwStores
        | KwTypes | KwFunctions | KwTransparent | KwMut | KwTransient | KwPersistent | KwNot
        | KwNeg | KwAdd | KwSub | KwMul | KwDiv | KwBitand | KwBitor | KwXor | KwShl | KwShr
        | KwSar | KwEq | KwNe | KwLt | KwLe | KwGt | KwGe | KwAnd | KwOr | KwTrue | KwFalse
        | KwNone => Some(KajitSemanticTokenType::Keyword),
        Colon | ColonColon | Comma | Eq | At | Amp | Minus | LAngle | RAngle => {
            Some(KajitSemanticTokenType::Operator)
        }
        Ident(name) => classify_ident(*name),
        LBrace | RBrace | LBracket | RBracket | LParen | RParen => None,
        RegionId(_) | StoreId(_) | ScopeId(_) | StmtId(_) => Some(KajitSemanticTokenType::Property),
    }
}

fn classify_ident(name: &str) -> Option<KajitSemanticTokenType> {
    if name.chars().next().is_some_and(char::is_uppercase) {
        return Some(KajitSemanticTokenType::Type);
    }
    Some(KajitSemanticTokenType::Property)
}

fn span_to_range(content: &str, span: HirSpan) -> Range {
    Range {
        start: offset_to_position(content, span.start),
        end: offset_to_position(content, span.end),
    }
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
        let source = r#"hir_module { functions [ function f0 params [] body { return "ok" } ] }"#;
        let tokens = compute_hir_semantic_tokens(source);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn reports_lexer_errors() {
        let source = "\"unterminated";
        let diagnostics = compute_hir_diagnostics(source);
        assert!(!diagnostics.is_empty());
    }
}
