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

/// Symbol definition with location information.
#[cfg(feature = "lsp")]
#[derive(Debug, Clone)]
struct SymbolDef {
    name: String,
    kind: SymbolKind,
    /// Byte offset range in source
    span_start: usize,
    span_end: usize,
    /// Line/character position (computed lazily)
    range: Option<Range>,
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
    /// Symbol definitions: URI -> list of definitions
    definitions: HashMap<String, Vec<SymbolDef>>,
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
            definitions: HashMap::new(),
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
                // Collect symbol definitions
                let definitions = self.collect_definitions(&source_file, content);

                let mut state = self.state.write().unwrap();
                state.ast_cache.insert(uri.to_string(), source_file.clone());
                state.definitions.insert(uri.to_string(), definitions);
                Some(source_file)
            }
            Err(_) => None,
        }
    }

    /// Convert byte offset to Position (line, character).
    fn byte_offset_to_position(content: &str, offset: usize) -> Position {
        let mut line = 0u32;
        let mut line_start = 0usize;

        for (i, ch) in content.char_indices() {
            if i >= offset {
                break;
            }
            if ch == '\n' {
                line += 1;
                line_start = i + 1;
            }
        }

        Position {
            line,
            character: (offset.saturating_sub(line_start)) as u32,
        }
    }

    /// Convert Position to byte offset.
    fn position_to_byte_offset(content: &str, position: Position) -> usize {
        let mut current_line = 0u32;
        let mut line_start = 0usize;

        for (i, ch) in content.char_indices() {
            if current_line == position.line {
                // Found the target line, add character offset
                let mut char_count = 0u32;
                for (j, c) in content[line_start..].char_indices() {
                    if char_count == position.character {
                        return line_start + j;
                    }
                    if c == '\n' {
                        break;
                    }
                    char_count += 1;
                }
                // Character is past end of line
                return line_start + content[line_start..].len().min(position.character as usize);
            }
            if ch == '\n' {
                current_line += 1;
                line_start = i + 1;
            }
        }

        // Position is past end of file
        content.len()
    }

    /// Get word at position, returning (word, start_offset, end_offset).
    fn get_word_at_position(content: &str, position: Position) -> Option<(String, usize, usize)> {
        let offset = Self::position_to_byte_offset(content, position);
        let bytes = content.as_bytes();

        if offset >= bytes.len() {
            return None;
        }

        // Find start of identifier
        let mut start = offset;
        while start > 0 {
            let ch = bytes[start - 1] as char;
            if ch.is_alphanumeric() || ch == '_' {
                start -= 1;
            } else {
                break;
            }
        }

        // Find end of identifier
        let mut end = offset;
        while end < bytes.len() {
            let ch = bytes[end] as char;
            if ch.is_alphanumeric() || ch == '_' {
                end += 1;
            } else {
                break;
            }
        }

        if start == end {
            return None;
        }

        let word = String::from_utf8_lossy(&bytes[start..end]).to_string();
        Some((word, start, end))
    }

    /// Collect symbol definitions from the AST.
    fn collect_definitions(&self, source_file: &crate::ast::SourceFile, content: &str) -> Vec<SymbolDef> {
        let mut defs = Vec::new();

        for spanned_item in &source_file.items {
            let item = &spanned_item.node;
            match item {
                crate::ast::Item::Function(func) => {
                    let span = &func.name.span;
                    let start_pos = Self::byte_offset_to_position(content, span.start);
                    let end_pos = Self::byte_offset_to_position(content, span.end);
                    defs.push(SymbolDef {
                        name: func.name.name.clone(),
                        kind: SymbolKind::FUNCTION,
                        span_start: span.start,
                        span_end: span.end,
                        range: Some(Range {
                            start: start_pos,
                            end: end_pos,
                        }),
                    });
                }
                crate::ast::Item::Struct(s) => {
                    let span = &s.name.span;
                    let start_pos = Self::byte_offset_to_position(content, span.start);
                    let end_pos = Self::byte_offset_to_position(content, span.end);
                    defs.push(SymbolDef {
                        name: s.name.name.clone(),
                        kind: SymbolKind::STRUCT,
                        span_start: span.start,
                        span_end: span.end,
                        range: Some(Range {
                            start: start_pos,
                            end: end_pos,
                        }),
                    });

                    // Also add struct fields
                    if let crate::ast::StructFields::Named(fields) = &s.fields {
                        for field in fields {
                            let field_span = &field.name.span;
                            let field_start = Self::byte_offset_to_position(content, field_span.start);
                            let field_end = Self::byte_offset_to_position(content, field_span.end);
                            defs.push(SymbolDef {
                                name: field.name.name.clone(),
                                kind: SymbolKind::FIELD,
                                span_start: field_span.start,
                                span_end: field_span.end,
                                range: Some(Range {
                                    start: field_start,
                                    end: field_end,
                                }),
                            });
                        }
                    }
                }
                crate::ast::Item::Enum(e) => {
                    let span = &e.name.span;
                    let start_pos = Self::byte_offset_to_position(content, span.start);
                    let end_pos = Self::byte_offset_to_position(content, span.end);
                    defs.push(SymbolDef {
                        name: e.name.name.clone(),
                        kind: SymbolKind::ENUM,
                        span_start: span.start,
                        span_end: span.end,
                        range: Some(Range {
                            start: start_pos,
                            end: end_pos,
                        }),
                    });

                    // Add enum variants
                    for variant in &e.variants {
                        let var_span = &variant.name.span;
                        let var_start = Self::byte_offset_to_position(content, var_span.start);
                        let var_end = Self::byte_offset_to_position(content, var_span.end);
                        defs.push(SymbolDef {
                            name: variant.name.name.clone(),
                            kind: SymbolKind::ENUM_MEMBER,
                            span_start: var_span.start,
                            span_end: var_span.end,
                            range: Some(Range {
                                start: var_start,
                                end: var_end,
                            }),
                        });
                    }
                }
                crate::ast::Item::Trait(t) => {
                    let span = &t.name.span;
                    let start_pos = Self::byte_offset_to_position(content, span.start);
                    let end_pos = Self::byte_offset_to_position(content, span.end);
                    defs.push(SymbolDef {
                        name: t.name.name.clone(),
                        kind: SymbolKind::INTERFACE,
                        span_start: span.start,
                        span_end: span.end,
                        range: Some(Range {
                            start: start_pos,
                            end: end_pos,
                        }),
                    });
                }
                crate::ast::Item::Impl(imp) => {
                    // Add methods from impl block
                    for item in &imp.items {
                        if let crate::ast::ImplItem::Function(method) = item {
                            let method_span = &method.name.span;
                            let method_start = Self::byte_offset_to_position(content, method_span.start);
                            let method_end = Self::byte_offset_to_position(content, method_span.end);
                            defs.push(SymbolDef {
                                name: method.name.name.clone(),
                                kind: SymbolKind::METHOD,
                                span_start: method_span.start,
                                span_end: method_span.end,
                                range: Some(Range {
                                    start: method_start,
                                    end: method_end,
                                }),
                            });
                        }
                    }
                }
                crate::ast::Item::Const(c) => {
                    let span = &c.name.span;
                    let start_pos = Self::byte_offset_to_position(content, span.start);
                    let end_pos = Self::byte_offset_to_position(content, span.end);
                    defs.push(SymbolDef {
                        name: c.name.name.clone(),
                        kind: SymbolKind::CONSTANT,
                        span_start: span.start,
                        span_end: span.end,
                        range: Some(Range {
                            start: start_pos,
                            end: end_pos,
                        }),
                    });
                }
                crate::ast::Item::Static(s) => {
                    let span = &s.name.span;
                    let start_pos = Self::byte_offset_to_position(content, span.start);
                    let end_pos = Self::byte_offset_to_position(content, span.end);
                    defs.push(SymbolDef {
                        name: s.name.name.clone(),
                        kind: SymbolKind::VARIABLE,
                        span_start: span.start,
                        span_end: span.end,
                        range: Some(Range {
                            start: start_pos,
                            end: end_pos,
                        }),
                    });
                }
                crate::ast::Item::TypeAlias(t) => {
                    let span = &t.name.span;
                    let start_pos = Self::byte_offset_to_position(content, span.start);
                    let end_pos = Self::byte_offset_to_position(content, span.end);
                    defs.push(SymbolDef {
                        name: t.name.name.clone(),
                        kind: SymbolKind::TYPE_PARAMETER,
                        span_start: span.start,
                        span_end: span.end,
                        range: Some(Range {
                            start: start_pos,
                            end: end_pos,
                        }),
                    });
                }
                _ => {}
            }
        }

        defs
    }

    /// Find all occurrences of a word in the document content.
    fn find_all_occurrences(content: &str, word: &str) -> Vec<Range> {
        let mut occurrences = Vec::new();
        let word_bytes = word.as_bytes();

        let mut search_start = 0usize;
        while let Some(pos) = content[search_start..].find(word) {
            let absolute_pos = search_start + pos;

            // Check word boundaries
            let before_ok = absolute_pos == 0
                || !content.as_bytes()[absolute_pos - 1].is_ascii_alphanumeric()
                    && content.as_bytes()[absolute_pos - 1] != b'_';

            let after_pos = absolute_pos + word_bytes.len();
            let after_ok = after_pos >= content.len()
                || !content.as_bytes()[after_pos].is_ascii_alphanumeric()
                    && content.as_bytes()[after_pos] != b'_';

            if before_ok && after_ok {
                let start = Self::byte_offset_to_position(content, absolute_pos);
                let end = Self::byte_offset_to_position(content, after_pos);
                occurrences.push(Range { start, end });
            }

            search_start = absolute_pos + 1;
        }

        occurrences
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

    /// Extract symbols from AST with proper source locations.
    fn extract_symbols(&self, source_file: &crate::ast::SourceFile, content: &str) -> Vec<DocumentSymbol> {
        let mut symbols = Vec::new();

        for spanned_item in &source_file.items {
            let item = &spanned_item.node;
            let item_span = &spanned_item.span;

            // Item range covers the entire declaration
            let item_start = Self::byte_offset_to_position(content, item_span.start);
            let item_end = Self::byte_offset_to_position(content, item_span.end);
            let item_range = Range {
                start: item_start,
                end: item_end,
            };

            match item {
                crate::ast::Item::Function(func) => {
                    // Selection range is just the name
                    let name_span = &func.name.span;
                    let name_start = Self::byte_offset_to_position(content, name_span.start);
                    let name_end = Self::byte_offset_to_position(content, name_span.end);
                    let selection_range = Range {
                        start: name_start,
                        end: name_end,
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: func.name.name.clone(),
                        detail: Some(format!("fn {}(...)", func.name.name)),
                        kind: SymbolKind::FUNCTION,
                        tags: None,
                        deprecated: None,
                        range: item_range,
                        selection_range,
                        children: None,
                    });
                }
                crate::ast::Item::Struct(s) => {
                    let name_span = &s.name.span;
                    let name_start = Self::byte_offset_to_position(content, name_span.start);
                    let name_end = Self::byte_offset_to_position(content, name_span.end);
                    let selection_range = Range {
                        start: name_start,
                        end: name_end,
                    };

                    // Collect field children
                    let children = if let crate::ast::StructFields::Named(fields) = &s.fields {
                        let field_symbols: Vec<DocumentSymbol> = fields
                            .iter()
                            .map(|field| {
                                let field_span = &field.name.span;
                                let field_start = Self::byte_offset_to_position(content, field_span.start);
                                let field_end = Self::byte_offset_to_position(content, field_span.end);
                                let field_range = Range {
                                    start: field_start,
                                    end: field_end,
                                };

                                #[allow(deprecated)]
                                DocumentSymbol {
                                    name: field.name.name.clone(),
                                    detail: None,
                                    kind: SymbolKind::FIELD,
                                    tags: None,
                                    deprecated: None,
                                    range: field_range,
                                    selection_range: field_range,
                                    children: None,
                                }
                            })
                            .collect();
                        if field_symbols.is_empty() {
                            None
                        } else {
                            Some(field_symbols)
                        }
                    } else {
                        None
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: s.name.name.clone(),
                        detail: Some(format!("struct {}", s.name.name)),
                        kind: SymbolKind::STRUCT,
                        tags: None,
                        deprecated: None,
                        range: item_range,
                        selection_range,
                        children,
                    });
                }
                crate::ast::Item::Enum(e) => {
                    let name_span = &e.name.span;
                    let name_start = Self::byte_offset_to_position(content, name_span.start);
                    let name_end = Self::byte_offset_to_position(content, name_span.end);
                    let selection_range = Range {
                        start: name_start,
                        end: name_end,
                    };

                    // Collect variant children
                    let children: Vec<DocumentSymbol> = e
                        .variants
                        .iter()
                        .map(|variant| {
                            let var_span = &variant.name.span;
                            let var_start = Self::byte_offset_to_position(content, var_span.start);
                            let var_end = Self::byte_offset_to_position(content, var_span.end);
                            let var_range = Range {
                                start: var_start,
                                end: var_end,
                            };

                            #[allow(deprecated)]
                            DocumentSymbol {
                                name: variant.name.name.clone(),
                                detail: None,
                                kind: SymbolKind::ENUM_MEMBER,
                                tags: None,
                                deprecated: None,
                                range: var_range,
                                selection_range: var_range,
                                children: None,
                            }
                        })
                        .collect();

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: e.name.name.clone(),
                        detail: Some(format!("enum {}", e.name.name)),
                        kind: SymbolKind::ENUM,
                        tags: None,
                        deprecated: None,
                        range: item_range,
                        selection_range,
                        children: if children.is_empty() {
                            None
                        } else {
                            Some(children)
                        },
                    });
                }
                crate::ast::Item::Trait(t) => {
                    let name_span = &t.name.span;
                    let name_start = Self::byte_offset_to_position(content, name_span.start);
                    let name_end = Self::byte_offset_to_position(content, name_span.end);
                    let selection_range = Range {
                        start: name_start,
                        end: name_end,
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: t.name.name.clone(),
                        detail: Some(format!("trait {}", t.name.name)),
                        kind: SymbolKind::INTERFACE,
                        tags: None,
                        deprecated: None,
                        range: item_range,
                        selection_range,
                        children: None,
                    });
                }
                crate::ast::Item::Impl(imp) => {
                    // Add methods from impl blocks
                    for item in &imp.items {
                        if let crate::ast::ImplItem::Function(method) = item {
                            let method_span = &method.name.span;
                            let method_start = Self::byte_offset_to_position(content, method_span.start);
                            let method_end = Self::byte_offset_to_position(content, method_span.end);
                            let method_range = Range {
                                start: method_start,
                                end: method_end,
                            };

                            #[allow(deprecated)]
                            symbols.push(DocumentSymbol {
                                name: method.name.name.clone(),
                                detail: Some(format!("fn {}(...)", method.name.name)),
                                kind: SymbolKind::METHOD,
                                tags: None,
                                deprecated: None,
                                range: method_range,
                                selection_range: method_range,
                                children: None,
                            });
                        }
                    }
                }
                crate::ast::Item::Const(c) => {
                    let name_span = &c.name.span;
                    let name_start = Self::byte_offset_to_position(content, name_span.start);
                    let name_end = Self::byte_offset_to_position(content, name_span.end);
                    let selection_range = Range {
                        start: name_start,
                        end: name_end,
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: c.name.name.clone(),
                        detail: Some("const".to_string()),
                        kind: SymbolKind::CONSTANT,
                        tags: None,
                        deprecated: None,
                        range: item_range,
                        selection_range,
                        children: None,
                    });
                }
                crate::ast::Item::Static(s) => {
                    let name_span = &s.name.span;
                    let name_start = Self::byte_offset_to_position(content, name_span.start);
                    let name_end = Self::byte_offset_to_position(content, name_span.end);
                    let selection_range = Range {
                        start: name_start,
                        end: name_end,
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: s.name.name.clone(),
                        detail: Some("static".to_string()),
                        kind: SymbolKind::VARIABLE,
                        tags: None,
                        deprecated: None,
                        range: item_range,
                        selection_range,
                        children: None,
                    });
                }
                crate::ast::Item::TypeAlias(t) => {
                    let name_span = &t.name.span;
                    let name_start = Self::byte_offset_to_position(content, name_span.start);
                    let name_end = Self::byte_offset_to_position(content, name_span.end);
                    let selection_range = Range {
                        start: name_start,
                        end: name_end,
                    };

                    #[allow(deprecated)]
                    symbols.push(DocumentSymbol {
                        name: t.name.name.clone(),
                        detail: Some("type".to_string()),
                        kind: SymbolKind::TYPE_PARAMETER,
                        tags: None,
                        deprecated: None,
                        range: item_range,
                        selection_range,
                        children: None,
                    });
                }
                _ => {}
            }
        }

        symbols
    }

    /// Get completion prefix (the word being typed before cursor).
    fn get_completion_prefix(content: &str, position: Position) -> String {
        let offset = Self::position_to_byte_offset(content, position);
        let bytes = content.as_bytes();

        if offset == 0 {
            return String::new();
        }

        let mut start = offset;
        while start > 0 {
            let ch = bytes[start - 1];
            if ch.is_ascii_alphanumeric() || ch == b'_' {
                start -= 1;
            } else {
                break;
            }
        }

        String::from_utf8_lossy(&bytes[start..offset]).to_string()
    }

    /// Get module-level completions (after · or :).
    fn get_module_completions(
        &self,
        _uri: &str,
        _position: Position,
        _state: &std::sync::RwLockReadGuard<ServerState>,
    ) -> Result<Option<CompletionResponse>> {
        // Standard library modules
        let modules = [
            ("new", "Constructor", CompletionItemKind::FUNCTION),
            ("from", "Create from value", CompletionItemKind::FUNCTION),
            ("default", "Default value", CompletionItemKind::FUNCTION),
            ("clone", "Clone value", CompletionItemKind::METHOD),
            ("len", "Get length", CompletionItemKind::METHOD),
            ("is_empty", "Check if empty", CompletionItemKind::METHOD),
            ("push", "Add element", CompletionItemKind::METHOD),
            ("pop", "Remove element", CompletionItemKind::METHOD),
            ("get", "Get by key", CompletionItemKind::METHOD),
            ("set", "Set by key", CompletionItemKind::METHOD),
            ("iter", "Get iterator", CompletionItemKind::METHOD),
            ("map", "Transform elements", CompletionItemKind::METHOD),
            ("filter", "Filter elements", CompletionItemKind::METHOD),
            ("fold", "Reduce to single value", CompletionItemKind::METHOD),
            ("unwrap", "Unwrap Option/Result", CompletionItemKind::METHOD),
            ("expect", "Unwrap with message", CompletionItemKind::METHOD),
            ("ok", "Convert to Option", CompletionItemKind::METHOD),
            ("err", "Get error", CompletionItemKind::METHOD),
            ("Some", "Option::Some variant", CompletionItemKind::ENUM_MEMBER),
            ("None", "Option::None variant", CompletionItemKind::ENUM_MEMBER),
            ("Ok", "Result::Ok variant", CompletionItemKind::ENUM_MEMBER),
            ("Err", "Result::Err variant", CompletionItemKind::ENUM_MEMBER),
        ];

        let items: Vec<CompletionItem> = modules
            .iter()
            .map(|(label, detail, kind)| CompletionItem {
                label: label.to_string(),
                kind: Some(*kind),
                detail: Some(detail.to_string()),
                ..Default::default()
            })
            .collect();

        Ok(Some(CompletionResponse::Array(items)))
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
                state
                    .versions
                    .insert(uri.clone(), params.text_document.version);
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
        let uri = params
            .text_document_position_params
            .text_document
            .uri
            .to_string();
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
        let uri = params.text_document_position.text_document.uri.to_string();
        let position = params.text_document_position.position;

        let state = self.state.read().unwrap();
        let content = state.documents.get(&uri);

        // Get the prefix being typed
        let prefix = if let Some(src) = content {
            Self::get_completion_prefix(src, position)
        } else {
            String::new()
        };

        let prefix_lower = prefix.to_lowercase();
        let mut items = Vec::new();

        // Check trigger character for context
        let trigger = params
            .context
            .as_ref()
            .and_then(|ctx| ctx.trigger_character.as_ref())
            .map(|s| s.as_str());

        // Handle module access (middledot · or colon :)
        if trigger == Some("·") || trigger == Some(":") {
            // Return module-level completions
            return self.get_module_completions(&uri, position, &state);
        }

        // Add document symbols first (higher priority)
        if let Some(definitions) = state.definitions.get(&uri) {
            for def in definitions {
                let name_lower = def.name.to_lowercase();
                if prefix.is_empty() || name_lower.starts_with(&prefix_lower) {
                    // Convert SymbolKind to CompletionItemKind
                    let completion_kind = match def.kind {
                        SymbolKind::FUNCTION => CompletionItemKind::FUNCTION,
                        SymbolKind::METHOD => CompletionItemKind::METHOD,
                        SymbolKind::STRUCT => CompletionItemKind::STRUCT,
                        SymbolKind::ENUM => CompletionItemKind::ENUM,
                        SymbolKind::ENUM_MEMBER => CompletionItemKind::ENUM_MEMBER,
                        SymbolKind::INTERFACE => CompletionItemKind::INTERFACE,
                        SymbolKind::CONSTANT => CompletionItemKind::CONSTANT,
                        SymbolKind::VARIABLE => CompletionItemKind::VARIABLE,
                        SymbolKind::FIELD => CompletionItemKind::FIELD,
                        SymbolKind::TYPE_PARAMETER => CompletionItemKind::TYPE_PARAMETER,
                        _ => CompletionItemKind::VALUE,
                    };

                    items.push(CompletionItem {
                        label: def.name.clone(),
                        kind: Some(completion_kind),
                        detail: Some(format!("(defined in this file)")),
                        sort_text: Some(format!("0_{}", def.name)), // Sort at top
                        ..Default::default()
                    });
                }
            }
        }

        // Keywords (Sigil-native vocabulary)
        let keywords = [
            // Core definitions
            ("rite", "Function definition (rite)", CompletionItemKind::KEYWORD),
            ("sigil", "Structure definition (sigil)", CompletionItemKind::KEYWORD),
            ("aspect", "Trait definition (aspect)", CompletionItemKind::KEYWORD),
            ("impl", "Implementation block", CompletionItemKind::KEYWORD),
            ("scroll", "Module definition (scroll)", CompletionItemKind::KEYWORD),
            ("invoke", "Import statement (invoke)", CompletionItemKind::KEYWORD),
            // Variable binding
            ("≔", "Variable binding (≔)", CompletionItemKind::KEYWORD),
            ("vary", "Mutable modifier (vary)", CompletionItemKind::KEYWORD),
            // Control flow
            ("if", "Conditional", CompletionItemKind::KEYWORD),
            ("else", "Else branch", CompletionItemKind::KEYWORD),
            ("match", "Pattern matching", CompletionItemKind::KEYWORD),
            ("while", "While loop", CompletionItemKind::KEYWORD),
            ("each", "For-each iteration (each)", CompletionItemKind::KEYWORD),
            ("forever", "Infinite loop (forever)", CompletionItemKind::KEYWORD),
            ("of", "Membership operator (of)", CompletionItemKind::KEYWORD),
            ("return", "Return statement", CompletionItemKind::KEYWORD),
            ("break", "Break loop", CompletionItemKind::KEYWORD),
            ("continue", "Continue loop", CompletionItemKind::KEYWORD),
            // Boolean literals
            ("yea", "Boolean true (yea)", CompletionItemKind::KEYWORD),
            ("nay", "Boolean false (nay)", CompletionItemKind::KEYWORD),
            ("true", "Boolean true", CompletionItemKind::KEYWORD),
            ("false", "Boolean false", CompletionItemKind::KEYWORD),
            // Visibility & modifiers
            ("☉", "Public visibility (☉)", CompletionItemKind::KEYWORD),
            ("async", "Async function", CompletionItemKind::KEYWORD),
            ("await", "Await expression", CompletionItemKind::KEYWORD),
            ("static", "Static variable", CompletionItemKind::KEYWORD),
            ("unsafe", "Unsafe block", CompletionItemKind::KEYWORD),
            // References
            ("this", "Self reference (this)", CompletionItemKind::KEYWORD),
            ("above", "Parent/super reference (above)", CompletionItemKind::KEYWORD),
            ("tome", "Crate reference (tome)", CompletionItemKind::KEYWORD),
            // Type system
            ("type", "Type alias", CompletionItemKind::KEYWORD),
            ("dyn", "Dynamic trait object", CompletionItemKind::KEYWORD),
            ("as", "Type cast", CompletionItemKind::KEYWORD),
            // Unicode alternatives
            ("λ", "Lambda function (λ)", CompletionItemKind::KEYWORD),
            ("→", "Return type arrow (→)", CompletionItemKind::KEYWORD),
            ("Σ", "Struct (Σ)", CompletionItemKind::KEYWORD),
        ];

        for (label, detail, kind) in keywords {
            let label_lower = label.to_lowercase();
            if prefix.is_empty() || label_lower.starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    label: label.to_string(),
                    kind: Some(kind),
                    detail: Some(detail.to_string()),
                    sort_text: Some(format!("1_{}", label)), // Sort after local symbols
                    ..Default::default()
                });
            }
        }

        // Built-in functions
        let functions = [
            ("println", "Print with newline", "fn println(value: Any)"),
            ("print", "Print without newline", "fn print(value: Any)"),
            ("len", "Get length", "fn len(collection: Array | String) -> i64"),
            ("push", "Append to array", "fn push(array: Array, value: Any)"),
            ("pop", "Remove last element", "fn pop(array: Array) -> Any"),
            ("get", "Get by index/key", "fn get(collection: Array | Map, key: Any) -> Any"),
            ("set", "Set by index/key", "fn set(collection: Array | Map, key: Any, value: Any)"),
            ("abs", "Absolute value", "fn abs(x: i64 | f64) -> i64 | f64"),
            ("sqrt", "Square root", "fn sqrt(x: f64) -> f64"),
            ("sin", "Sine", "fn sin(x: f64) -> f64"),
            ("cos", "Cosine", "fn cos(x: f64) -> f64"),
            ("floor", "Round down", "fn floor(x: f64) -> f64"),
            ("ceil", "Round up", "fn ceil(x: f64) -> f64"),
            ("upper", "To uppercase", "fn upper(s: String) -> String"),
            ("lower", "To lowercase", "fn lower(s: String) -> String"),
            ("trim", "Trim whitespace", "fn trim(s: String) -> String"),
            ("split", "Split string", "fn split(s: String, delim: String) -> Array<String>"),
            ("replace", "Replace substring", "fn replace(s: String, from: String, to: String) -> String"),
            ("type_of", "Get type name", "fn type_of(value: Any) -> String"),
            ("to_string", "Convert to string", "fn to_string(value: Any) -> String"),
            ("parse_int", "Parse integer", "fn parse_int(s: String) -> i64"),
            ("parse_float", "Parse float", "fn parse_float(s: String) -> f64"),
        ];

        for (label, detail, signature) in functions {
            let label_lower = label.to_lowercase();
            if prefix.is_empty() || label_lower.starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    label: label.to_string(),
                    kind: Some(CompletionItemKind::FUNCTION),
                    detail: Some(detail.to_string()),
                    documentation: Some(Documentation::String(signature.to_string())),
                    sort_text: Some(format!("2_{}", label)),
                    ..Default::default()
                });
            }
        }

        // Types
        let types = [
            ("i64", "64-bit signed integer"),
            ("i32", "32-bit signed integer"),
            ("i16", "16-bit signed integer"),
            ("i8", "8-bit signed integer"),
            ("u64", "64-bit unsigned integer"),
            ("u32", "32-bit unsigned integer"),
            ("u16", "16-bit unsigned integer"),
            ("u8", "8-bit unsigned integer"),
            ("f64", "64-bit floating point"),
            ("f32", "32-bit floating point"),
            ("bool", "Boolean type"),
            ("String", "UTF-8 string type"),
            ("char", "Unicode character"),
            ("Array", "Dynamic array"),
            ("Vec", "Dynamic vector (alias for Array)"),
            ("Map", "Hash map"),
            ("Option", "Optional value (Some/None)"),
            ("Result", "Result type (Ok/Err)"),
            ("Rc", "Reference counted pointer"),
            ("Cell", "Interior mutability cell"),
            ("Box", "Heap-allocated pointer"),
        ];

        for (label, detail) in types {
            let label_lower = label.to_lowercase();
            if prefix.is_empty() || label_lower.starts_with(&prefix_lower) {
                items.push(CompletionItem {
                    label: label.to_string(),
                    kind: Some(CompletionItemKind::TYPE_PARAMETER),
                    detail: Some(detail.to_string()),
                    sort_text: Some(format!("3_{}", label)),
                    ..Default::default()
                });
            }
        }

        // Snippets for common patterns (Sigil syntax)
        if prefix.is_empty() || "rite".starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: "rite (function)".to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                insert_text: Some("rite ${1:name}(${2:params}) → ${3:!ReturnType} {\n\t$0\n}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                detail: Some("Function definition (rite)".to_string()),
                sort_text: Some("4_rite".to_string()),
                ..Default::default()
            });
        }

        if prefix.is_empty() || "sigil".starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: "sigil (struct)".to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                insert_text: Some("sigil ${1:Name} {\n\t${2:field}: !${3:Type},\n}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                detail: Some("Structure definition (sigil)".to_string()),
                sort_text: Some("4_sigil".to_string()),
                ..Default::default()
            });
        }

        if prefix.is_empty() || "impl".starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: "impl (block)".to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                insert_text: Some("impl ${1:Type} {\n\t$0\n}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                detail: Some("Implementation block".to_string()),
                sort_text: Some("4_impl".to_string()),
                ..Default::default()
            });
        }

        if prefix.is_empty() || "match".starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: "match (expression)".to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                insert_text: Some("match ${1:value} {\n\t${2:pattern} => ${3:expr},\n\t_ => ${0:default},\n}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                detail: Some("Match expression".to_string()),
                sort_text: Some("4_match".to_string()),
                ..Default::default()
            });
        }

        if prefix.is_empty() || "aspect".starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: "aspect (trait)".to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                insert_text: Some("aspect ${1:Name} {\n\trite ${2:method}(&this) → ${3:!ReturnType};\n}".to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                detail: Some("Trait definition (aspect)".to_string()),
                sort_text: Some("4_aspect".to_string()),
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

        // Need both AST and content for symbol extraction
        let program = state.ast_cache.get(&uri);
        let content = state.documents.get(&uri);

        match (program, content) {
            (Some(ast), Some(src)) => {
                let symbols = self.extract_symbols(ast, src);
                Ok(Some(DocumentSymbolResponse::Nested(symbols)))
            }
            _ => Ok(None),
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
        let uri = params
            .text_document_position_params
            .text_document
            .uri
            .to_string();
        let position = params.text_document_position_params.position;

        let state = self.state.read().unwrap();

        // Get document content
        let content = match state.documents.get(&uri) {
            Some(c) => c,
            None => return Ok(None),
        };

        // Get word at cursor position
        let (word, _, _) = match Self::get_word_at_position(content, position) {
            Some(w) => w,
            None => return Ok(None),
        };

        // Look up definition in symbol table
        if let Some(definitions) = state.definitions.get(&uri) {
            for def in definitions {
                if def.name == word {
                    if let Some(range) = def.range {
                        let location = Location {
                            uri: Url::parse(&uri).unwrap_or_else(|_| {
                                params.text_document_position_params.text_document.uri.clone()
                            }),
                            range,
                        };
                        return Ok(Some(GotoDefinitionResponse::Scalar(location)));
                    }
                }
            }
        }

        Ok(None)
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = params
            .text_document_position
            .text_document
            .uri
            .to_string();
        let position = params.text_document_position.position;

        let state = self.state.read().unwrap();

        // Get document content
        let content = match state.documents.get(&uri) {
            Some(c) => c,
            None => return Ok(None),
        };

        // Get word at cursor position
        let (word, _, _) = match Self::get_word_at_position(content, position) {
            Some(w) => w,
            None => return Ok(None),
        };

        // Find all occurrences of the word
        let occurrences = Self::find_all_occurrences(content, &word);

        if occurrences.is_empty() {
            return Ok(None);
        }

        let parsed_uri = Url::parse(&uri).unwrap_or_else(|_| {
            params.text_document_position.text_document.uri.clone()
        });

        let locations: Vec<Location> = occurrences
            .into_iter()
            .map(|range| Location {
                uri: parsed_uri.clone(),
                range,
            })
            .collect();

        Ok(Some(locations))
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        let query = params.query.to_lowercase();
        let state = self.state.read().unwrap();
        let mut results = Vec::new();

        // Search through all open documents
        for (uri, definitions) in &state.definitions {
            for def in definitions {
                // Filter by query (empty query matches all)
                if query.is_empty() || def.name.to_lowercase().contains(&query) {
                    if let Some(range) = def.range {
                        let parsed_uri = match Url::parse(uri) {
                            Ok(u) => u,
                            Err(_) => continue,
                        };

                        #[allow(deprecated)]
                        results.push(SymbolInformation {
                            name: def.name.clone(),
                            kind: def.kind,
                            tags: None,
                            deprecated: None,
                            location: Location {
                                uri: parsed_uri,
                                range,
                            },
                            container_name: None,
                        });
                    }
                }
            }
        }

        if results.is_empty() {
            Ok(None)
        } else {
            Ok(Some(results))
        }
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
