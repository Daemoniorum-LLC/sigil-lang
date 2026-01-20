//! LSP Server for Sigil Language
//!
//! Provides IDE features via the Language Server Protocol:
//! - Real-time diagnostics (errors, warnings, lints)
//! - Hover information (types, documentation)
//! - Go to definition
//! - Find references
//! - Code completion
//! - Code actions (quick fixes)
//! - Document symbols
//! - Workspace symbols

#[cfg(feature = "lsp")]
use tower_lsp::jsonrpc::Result;
#[cfg(feature = "lsp")]
use tower_lsp::lsp_types::*;
#[cfg(feature = "lsp")]
use tower_lsp::{Client, LanguageServer, LspService, Server};

use crate::lint::{lint_for_lsp, LintConfig};
use crate::parser::Parser;
use std::collections::HashMap;
use std::sync::RwLock;

/// Sigil Language Server
#[cfg(feature = "lsp")]
pub struct SigilLanguageServer {
    client: Client,
    state: RwLock<ServerState>,
}

/// Internal server state
#[cfg(feature = "lsp")]
struct ServerState {
    /// Open documents: URI -> content
    documents: HashMap<String, String>,
    /// Document versions
    versions: HashMap<String, i32>,
    /// Lint configuration
    lint_config: LintConfig,
    /// Parsed ASTs cache
    ast_cache: HashMap<String, crate::ast::SourceFile>,
    /// Symbol table cache
    symbols: HashMap<String, Vec<DocumentSymbol>>,
}

#[cfg(feature = "lsp")]
impl ServerState {
    fn new() -> Self {
        Self {
            documents: HashMap::new(),
            versions: HashMap::new(),
            lint_config: LintConfig::find_and_load(),
            ast_cache: HashMap::new(),
            symbols: HashMap::new(),
        }
    }
}

#[cfg(feature = "lsp")]
impl SigilLanguageServer {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            state: RwLock::new(ServerState::new()),
        }
    }

    /// Publish diagnostics for a document
    async fn publish_diagnostics(&self, uri: Url) {
        let diagnostics = {
            let state = self.state.read().unwrap();
            let uri_str = uri.to_string();

            if let Some(content) = state.documents.get(&uri_str) {
                let lsp_result = lint_for_lsp(content, &uri_str, state.lint_config.clone());

                // Convert our LspDiagnostic to tower-lsp Diagnostic
                lsp_result
                    .diagnostics
                    .into_iter()
                    .map(|d| Diagnostic {
                        range: Range {
                            start: Position {
                                line: d.line,
                                character: d.character,
                            },
                            end: Position {
                                line: d.end_line,
                                character: d.end_character,
                            },
                        },
                        severity: Some(match d.severity {
                            1 => DiagnosticSeverity::ERROR,
                            2 => DiagnosticSeverity::WARNING,
                            3 => DiagnosticSeverity::INFORMATION,
                            _ => DiagnosticSeverity::HINT,
                        }),
                        code: d.code.map(NumberOrString::String),
                        source: Some(d.source),
                        message: d.message,
                        related_information: if d.related_information.is_empty() {
                            None
                        } else {
                            Some(
                                d.related_information
                                    .into_iter()
                                    .map(|r| DiagnosticRelatedInformation {
                                        location: Location {
                                            uri: Url::parse(&r.uri).unwrap_or_else(|_| uri.clone()),
                                            range: Range {
                                                start: Position {
                                                    line: r.line,
                                                    character: r.character,
                                                },
                                                end: Position {
                                                    line: r.line,
                                                    character: r.character + 1,
                                                },
                                            },
                                        },
                                        message: r.message,
                                    })
                                    .collect(),
                            )
                        },
                        ..Default::default()
                    })
                    .collect()
            } else {
                vec![]
            }
        };

        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }

    /// Parse document and cache AST
    fn parse_document(&self, uri: &str, content: &str) -> Option<crate::ast::SourceFile> {
        let mut parser = Parser::new(content);

        match parser.parse_file() {
            Ok(source_file) => {
                let mut state = self.state.write().unwrap();
                state.ast_cache.insert(uri.to_string(), source_file.clone());
                Some(source_file)
            }
            Err(_) => None,
        }
    }

    /// Get hover information for a position
    fn get_hover_info(&self, uri: &str, position: Position) -> Option<String> {
        let state = self.state.read().unwrap();
        let content = state.documents.get(uri)?;

        // Find the word at position
        let lines: Vec<&str> = content.lines().collect();
        if position.line as usize >= lines.len() {
            return None;
        }

        let line = lines[position.line as usize];
        let char_pos = position.character as usize;

        if char_pos >= line.len() {
            return None;
        }

        // Extract identifier at position
        let chars: Vec<char> = line.chars().collect();
        let mut start = char_pos;
        let mut end = char_pos;

        // Find start of identifier
        while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '_') {
            start -= 1;
        }

        // Find end of identifier
        while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '_') {
            end += 1;
        }

        let word: String = chars[start..end].iter().collect();

        if word.is_empty() {
            return None;
        }

        // Check for built-in functions and types
        self.get_builtin_docs(&word)
    }

    /// Get documentation for built-in functions and types
    fn get_builtin_docs(&self, name: &str) -> Option<String> {
        match name {
            // Core types
            "i64" => Some("**i64** - 64-bit signed integer\n\nRange: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807".to_string()),
            "f64" => Some("**f64** - 64-bit floating point number\n\nIEEE 754 double precision".to_string()),
            "bool" => Some("**bool** - Boolean type\n\nValues: `true` or `false`".to_string()),
            "String" => Some("**String** - UTF-8 string type\n\nImmutable string of characters".to_string()),

            // Control flow keywords
            "if" => Some("**if** - Conditional expression\n\n```sigil\nif condition {\n    // then branch\n} else {\n    // else branch\n}\n```".to_string()),
            "while" => Some("**while** - Loop while condition is true\n\n```sigil\nwhile condition {\n    // loop body\n}\n```".to_string()),
            "for" => Some("**for** - Iterate over a range or collection\n\n```sigil\nfor i in 0..10 {\n    println(i);\n}\n```".to_string()),
            "match" => Some("**match** - Pattern matching expression\n\n```sigil\nmatch value {\n    pattern1 => expr1,\n    pattern2 => expr2,\n    _ => default\n}\n```".to_string()),
            "return" => Some("**return** - Return from function\n\n```sigil\nreturn value;\n```".to_string()),

            // Common functions
            "println" => Some("**println**(value: Any) -> ()\n\nPrint a value followed by newline to stdout.".to_string()),
            "print" => Some("**print**(value: Any) -> ()\n\nPrint a value to stdout without newline.".to_string()),
            "len" => Some("**len**(collection: Array | String) -> i64\n\nReturn the length of an array or string.".to_string()),
            "push" => Some("**push**(array: Array, value: Any) -> ()\n\nAppend a value to the end of an array.".to_string()),
            "pop" => Some("**pop**(array: Array) -> Any\n\nRemove and return the last element of an array.".to_string()),
            "get" => Some("**get**(collection: Array | Map, key: Any) -> Any\n\nGet a value from an array by index or map by key.".to_string()),
            "set" => Some("**set**(collection: Array | Map, key: Any, value: Any) -> ()\n\nSet a value in an array or map.".to_string()),

            // Math functions
            "abs" => Some("**abs**(x: i64 | f64) -> i64 | f64\n\nReturn the absolute value.".to_string()),
            "sqrt" => Some("**sqrt**(x: f64) -> f64\n\nReturn the square root.".to_string()),
            "sin" => Some("**sin**(x: f64) -> f64\n\nReturn the sine of x (in radians).".to_string()),
            "cos" => Some("**cos**(x: f64) -> f64\n\nReturn the cosine of x (in radians).".to_string()),
            "floor" => Some("**floor**(x: f64) -> f64\n\nRound down to nearest integer.".to_string()),
            "ceil" => Some("**ceil**(x: f64) -> f64\n\nRound up to nearest integer.".to_string()),

            // String functions
            "upper" => Some("**upper**(s: String) -> String\n\nConvert string to uppercase.".to_string()),
            "lower" => Some("**lower**(s: String) -> String\n\nConvert string to lowercase.".to_string()),
            "trim" => Some("**trim**(s: String) -> String\n\nRemove leading and trailing whitespace.".to_string()),
            "split" => Some("**split**(s: String, delimiter: String) -> Array<String>\n\nSplit string by delimiter.".to_string()),
            "replace" => Some("**replace**(s: String, from: String, to: String) -> String\n\nReplace all occurrences of `from` with `to`.".to_string()),

            // Type functions
            "type_of" => Some("**type_of**(value: Any) -> String\n\nReturn the type name of a value.".to_string()),
            "to_string" => Some("**to_string**(value: Any) -> String\n\nConvert a value to its string representation.".to_string()),
            "parse_int" => Some("**parse_int**(s: String) -> i64\n\nParse a string as an integer.".to_string()),
            "parse_float" => Some("**parse_float**(s: String) -> f64\n\nParse a string as a floating point number.".to_string()),

            // Evidence/trust keywords
            "trusted" => Some("**trusted** - Evidence level indicating verified/trusted data\n\nData that has been validated or comes from a trusted source.".to_string()),
            "untrusted" => Some("**untrusted** - Evidence level indicating unverified data\n\nData that has not been validated, such as user input.".to_string()),
            "tainted" => Some("**tainted** - Evidence level indicating potentially dangerous data\n\nData that may contain malicious content.".to_string()),

            // Modifiers
            "mut" => Some("**mut** - Mutable binding modifier\n\n```sigil\nlet mut x = 0;\nx = x + 1;  // OK: x is mutable\n```".to_string()),
            "let" => Some("**let** - Variable binding\n\n```sigil\nlet x = 42;        // immutable\nlet mut y = 0;     // mutable\n```".to_string()),
            "fn" => Some("**fn** - Function definition\n\n```sigil\nfn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n```".to_string()),
            "struct" => Some("**struct** - Structure definition\n\n```sigil\nstruct Point {\n    x: f64,\n    y: f64,\n}\n```".to_string()),
            "enum" => Some("**enum** - Enumeration definition\n\n```sigil\nenum Option<T> {\n    Some(T),\n    None,\n}\n```".to_string()),
            "trait" => Some("**trait** - Trait definition\n\n```sigil\ntrait Display {\n    fn display(&self) -> String;\n}\n```".to_string()),
            "impl" => Some("**impl** - Implementation block\n\n```sigil\nimpl Point {\n    fn new(x: f64, y: f64) -> Point {\n        Point { x, y }\n    }\n}\n```".to_string()),

            _ => None,
        }
    }

    /// Extract symbols from AST
    fn extract_symbols(&self, source_file: &crate::ast::SourceFile) -> Vec<DocumentSymbol> {
        let mut symbols = Vec::new();

        for spanned_item in &source_file.items {
            let item = &spanned_item.node;
            match item {
                crate::ast::Item::Function(func) => {
                    let range = Range {
                        start: Position { line: 0, character: 0 },
                        end: Position { line: 0, character: 0 },
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: func.name.name.clone(),
                        detail: Some(format!("fn {}(...)", func.name.name)),
                        kind: SymbolKind::FUNCTION,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Struct(s) => {
                    let range = Range {
                        start: Position { line: 0, character: 0 },
                        end: Position { line: 0, character: 0 },
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: s.name.name.clone(),
                        detail: Some(format!("struct {}", s.name.name)),
                        kind: SymbolKind::STRUCT,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Enum(e) => {
                    let range = Range {
                        start: Position { line: 0, character: 0 },
                        end: Position { line: 0, character: 0 },
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: e.name.name.clone(),
                        detail: Some(format!("enum {}", e.name.name)),
                        kind: SymbolKind::ENUM,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                crate::ast::Item::Trait(t) => {
                    let range = Range {
                        start: Position { line: 0, character: 0 },
                        end: Position { line: 0, character: 0 },
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: t.name.name.clone(),
                        detail: Some(format!("trait {}", t.name.name)),
                        kind: SymbolKind::INTERFACE,
                        tags: None,
                        deprecated: None,
                        range,
                        selection_range: range,
                        children: None,
                    });
                }
                _ => {}
            }
        }

        symbols
    }
}

#[cfg(feature = "lsp")]
#[tower_lsp::async_trait]
impl LanguageServer for SigilLanguageServer {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
                    resolve_provider: Some(false),
                    ..Default::default()
                }),
                document_symbol_provider: Some(OneOf::Left(true)),
                workspace_symbol_provider: Some(OneOf::Left(true)),
                code_action_provider: Some(CodeActionProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "sigil-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Sigil LSP server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        let content = params.text_document.text;
        let version = params.text_document.version;

        {
            let mut state = self.state.write().unwrap();
            state.documents.insert(uri.clone(), content.clone());
            state.versions.insert(uri.clone(), version);
        }

        // Parse and cache AST
        self.parse_document(&uri, &content);

        // Publish diagnostics
        self.publish_diagnostics(params.text_document.uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.to_string();

        // Get the full content (we use FULL sync)
        if let Some(change) = params.content_changes.into_iter().next() {
            let content = change.text;

            {
                let mut state = self.state.write().unwrap();
                state.documents.insert(uri.clone(), content.clone());
                state.versions.insert(uri.clone(), params.text_document.version);
            }

            // Re-parse and publish diagnostics
            self.parse_document(&uri, &content);
            self.publish_diagnostics(params.text_document.uri).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri.to_string();

        let mut state = self.state.write().unwrap();
        state.documents.remove(&uri);
        state.versions.remove(&uri);
        state.ast_cache.remove(&uri);
        state.symbols.remove(&uri);
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri.to_string();
        let position = params.text_document_position_params.position;

        if let Some(info) = self.get_hover_info(&uri, position) {
            Ok(Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: info,
                }),
                range: None,
            }))
        } else {
            Ok(None)
        }
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let mut items = Vec::new();

        // Keywords
        let keywords = [
            ("fn", "Function definition", CompletionItemKind::KEYWORD),
            ("let", "Variable binding", CompletionItemKind::KEYWORD),
            ("mut", "Mutable modifier", CompletionItemKind::KEYWORD),
            ("if", "Conditional", CompletionItemKind::KEYWORD),
            ("else", "Else branch", CompletionItemKind::KEYWORD),
            ("while", "While loop", CompletionItemKind::KEYWORD),
            ("for", "For loop", CompletionItemKind::KEYWORD),
            ("in", "Iterator keyword", CompletionItemKind::KEYWORD),
            ("match", "Pattern match", CompletionItemKind::KEYWORD),
            ("return", "Return statement", CompletionItemKind::KEYWORD),
            ("struct", "Structure definition", CompletionItemKind::KEYWORD),
            ("enum", "Enumeration definition", CompletionItemKind::KEYWORD),
            ("trait", "Trait definition", CompletionItemKind::KEYWORD),
            ("impl", "Implementation block", CompletionItemKind::KEYWORD),
            ("true", "Boolean true", CompletionItemKind::KEYWORD),
            ("false", "Boolean false", CompletionItemKind::KEYWORD),
        ];

        for (label, detail, kind) in keywords {
            items.push(CompletionItem {
                label: label.to_string(),
                kind: Some(kind),
                detail: Some(detail.to_string()),
                ..Default::default()
            });
        }

        // Built-in functions
        let functions = [
            ("println", "Print with newline"),
            ("print", "Print without newline"),
            ("len", "Get length"),
            ("push", "Append to array"),
            ("pop", "Remove last element"),
            ("get", "Get by index/key"),
            ("set", "Set by index/key"),
            ("abs", "Absolute value"),
            ("sqrt", "Square root"),
            ("sin", "Sine"),
            ("cos", "Cosine"),
            ("floor", "Round down"),
            ("ceil", "Round up"),
            ("upper", "To uppercase"),
            ("lower", "To lowercase"),
            ("trim", "Trim whitespace"),
            ("split", "Split string"),
            ("replace", "Replace substring"),
            ("type_of", "Get type name"),
            ("to_string", "Convert to string"),
        ];

        for (label, detail) in functions {
            items.push(CompletionItem {
                label: label.to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some(detail.to_string()),
                ..Default::default()
            });
        }

        // Types
        let types = ["i64", "f64", "bool", "String", "Array", "Map", "Option", "Result"];

        for ty in types {
            items.push(CompletionItem {
                label: ty.to_string(),
                kind: Some(CompletionItemKind::TYPE_PARAMETER),
                ..Default::default()
            });
        }

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = params.text_document.uri.to_string();

        let state = self.state.read().unwrap();

        if let Some(program) = state.ast_cache.get(&uri) {
            let symbols = self.extract_symbols(program);
            Ok(Some(DocumentSymbolResponse::Nested(symbols)))
        } else {
            Ok(None)
        }
    }

    async fn code_action(&self, params: CodeActionParams) -> Result<Option<CodeActionResponse>> {
        let uri = params.text_document.uri.to_string();
        let mut actions = Vec::new();

        let state = self.state.read().unwrap();

        if let Some(content) = state.documents.get(&uri) {
            let lsp_result = lint_for_lsp(content, &uri, state.lint_config.clone());

            // Find diagnostics in the requested range and create code actions
            for diag in lsp_result.diagnostics {
                // Check if diagnostic overlaps with requested range
                let diag_range = Range {
                    start: Position {
                        line: diag.line,
                        character: diag.character,
                    },
                    end: Position {
                        line: diag.end_line,
                        character: diag.end_character,
                    },
                };

                if ranges_overlap(&diag_range, &params.range) {
                    for action in diag.code_actions {
                        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                            title: action.title,
                            kind: Some(CodeActionKind::QUICKFIX),
                            diagnostics: None,
                            edit: Some(WorkspaceEdit {
                                changes: Some({
                                    let mut changes = HashMap::new();
                                    changes.insert(
                                        params.text_document.uri.clone(),
                                        vec![TextEdit {
                                            range: Range {
                                                start: Position {
                                                    line: action.edit.line,
                                                    character: action.edit.character,
                                                },
                                                end: Position {
                                                    line: action.edit.end_line,
                                                    character: action.edit.end_character,
                                                },
                                            },
                                            new_text: action.edit.new_text,
                                        }],
                                    );
                                    changes
                                }),
                                ..Default::default()
                            }),
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        if actions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(actions))
        }
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        // TODO: Implement proper go-to-definition with symbol resolution
        // For now, return None
        Ok(None)
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        // TODO: Implement proper find references with symbol resolution
        // For now, return None
        Ok(None)
    }
}

/// Check if two ranges overlap
#[cfg(feature = "lsp")]
fn ranges_overlap(a: &Range, b: &Range) -> bool {
    !(a.end.line < b.start.line
        || (a.end.line == b.start.line && a.end.character < b.start.character)
        || b.end.line < a.start.line
        || (b.end.line == a.start.line && b.end.character < a.start.character))
}

/// Run the LSP server
#[cfg(feature = "lsp")]
pub async fn run_lsp_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| SigilLanguageServer::new(client));
    Server::new(stdin, stdout, socket).serve(service).await;
}

/// Entry point for LSP command (called from main)
#[cfg(feature = "lsp")]
pub fn start_lsp() -> std::process::ExitCode {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(run_lsp_server());
    std::process::ExitCode::SUCCESS
}

#[cfg(not(feature = "lsp"))]
pub fn start_lsp() -> std::process::ExitCode {
    eprintln!("Error: LSP server not available (compile with --features lsp)");
    std::process::ExitCode::from(1)
}
