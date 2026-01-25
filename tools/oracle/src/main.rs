//! Oracle - Language Server Protocol implementation for Sigil
//!
//! Provides IDE features for the Sigil programming language:
//! - Real-time diagnostics (parse errors, type errors)
//! - Hover information (types, documentation)
//! - Go-to-definition
//! - Completions
//!
//! Designed for both human developers and AI agents.

use dashmap::DashMap;
use ropey::Rope;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use tracing::info;

use sigil_parser::{Lexer, Parser, Token, Span, TypeChecker};
use std::collections::HashMap;

/// Check if a character can continue a Unicode identifier.
/// This covers common Unicode letters and combining marks used in identifiers.
fn is_xid_continue(c: char) -> bool {
    // Unicode categories that can continue an identifier:
    // - Lu (uppercase letters), Ll (lowercase letters), Lt (titlecase)
    // - Lm (modifier letters), Lo (other letters)
    // - Mn (non-spacing marks), Mc (spacing combining marks)
    // - Nd (decimal numbers), Pc (connector punctuation like _)
    // - Nl (letter numbers like Roman numerals)
    // This is a simplified check - full XID_Continue is more complex
    c.is_alphabetic() || c.is_numeric() || c == '_' ||
    matches!(c, '\u{0300}'..='\u{036F}' | // Combining diacritical marks
             '\u{0370}'..='\u{03FF}' | // Greek (τ, φ, σ, ρ, λ, etc.)
             '\u{2200}'..='\u{22FF}' | // Mathematical operators (∀, ∃, etc.)
             '\u{2070}'..='\u{209F}')   // Superscripts and subscripts
}

/// A symbol definition in the document.
#[derive(Debug, Clone)]
struct SymbolDef {
    name: String,
    kind: SymbolKind,
    span: Span,
    #[allow(dead_code)]
    detail: Option<String>,
}

/// Semantic token types for highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SemanticTokenType {
    Keyword,
    Function,
    Variable,
    Parameter,
    Type,
    Morpheme,
    Evidentiality,
    Operator,
    Comment,
    String,
    Number,
}

impl SemanticTokenType {
    fn to_index(self) -> u32 {
        match self {
            Self::Keyword => 0,
            Self::Function => 1,
            Self::Variable => 2,
            Self::Parameter => 3,
            Self::Type => 4,
            Self::Morpheme => 5,
            Self::Evidentiality => 6,
            Self::Operator => 7,
            Self::Comment => 8,
            Self::String => 9,
            Self::Number => 10,
        }
    }

    fn legend() -> Vec<SemanticTokenType> {
        vec![
            Self::Keyword,
            Self::Function,
            Self::Variable,
            Self::Parameter,
            Self::Type,
            Self::Morpheme,
            Self::Evidentiality,
            Self::Operator,
            Self::Comment,
            Self::String,
            Self::Number,
        ]
    }
}

/// Document state tracked by the server.
struct Document {
    /// The document content as a rope for efficient editing.
    content: Rope,
    /// The document version.
    version: i32,
    /// Symbol definitions in the document.
    symbols: Vec<SymbolDef>,
    /// Symbol references (name -> list of usage spans).
    references: HashMap<String, Vec<Span>>,
    /// Type information for symbols (name -> type string).
    types: HashMap<String, String>,
}

/// The Oracle language server.
struct OracleServer {
    /// LSP client for sending notifications.
    client: Client,
    /// Open documents indexed by URI.
    documents: DashMap<Url, Document>,
}

impl OracleServer {
    fn new(client: Client) -> Self {
        Self {
            client,
            documents: DashMap::new(),
        }
    }

    /// Analyze a document and return diagnostics.
    async fn analyze_document(&self, uri: &Url) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        let content = match self.documents.get(uri) {
            Some(doc) => doc.content.to_string(),
            None => return diagnostics,
        };

        // Parse the document
        let mut parser = Parser::new(&content);
        match parser.parse_file() {
            Ok(ast) => {
                // Run type checker and collect type errors
                let mut type_checker = TypeChecker::new();
                if let Err(type_errors) = type_checker.check_file(&ast) {
                    for err in type_errors {
                        let (line, col, end_col) = self.span_to_position(&content, err.span);

                        let mut message = err.message.clone();
                        if !err.notes.is_empty() {
                            message.push_str("\n\nNotes:\n");
                            for note in &err.notes {
                                message.push_str("  • ");
                                message.push_str(note);
                                message.push('\n');
                            }
                        }

                        diagnostics.push(Diagnostic {
                            range: Range {
                                start: Position { line, character: col },
                                end: Position { line, character: end_col },
                            },
                            severity: Some(DiagnosticSeverity::ERROR),
                            code: Some(NumberOrString::String("type-error".to_string())),
                            source: Some("sigil-oracle".to_string()),
                            message,
                            related_information: None,
                            tags: None,
                            code_description: None,
                            data: None,
                        });
                    }
                }
            }
            Err(e) => {
                // Convert parse error to diagnostic
                let (line, col, end_col) = self.span_to_position(&content, e.span());

                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position { line, character: col },
                        end: Position { line, character: end_col },
                    },
                    severity: Some(DiagnosticSeverity::ERROR),
                    code: Some(NumberOrString::String("parse-error".to_string())),
                    source: Some("sigil-oracle".to_string()),
                    message: e.to_string(),
                    related_information: None,
                    tags: None,
                    code_description: None,
                    data: None,
                });
            }
        }

        diagnostics
    }

    /// Convert a byte span to line/column positions.
    fn span_to_position(&self, content: &str, span: Option<Span>) -> (u32, u32, u32) {
        let span = span.unwrap_or(Span::new(0, 1));
        let mut line = 0u32;
        let mut col = 0u32;
        let mut end_col = col;

        for (i, ch) in content.char_indices() {
            if i >= span.start {
                if i >= span.end {
                    end_col = col;
                    break;
                }
                if i == span.start {
                    // Found start position
                }
            }
            if ch == '\n' {
                if i < span.start {
                    line += 1;
                    col = 0;
                } else if i < span.end {
                    end_col = col;
                }
            } else {
                if i < span.start {
                    col += 1;
                }
                if i >= span.start && i < span.end {
                    end_col = col + 1;
                }
            }
        }

        (line, col, end_col.max(col + 1))
    }

    /// Get the token at a position in the document.
    fn get_token_at_position(&self, content: &str, position: Position) -> Option<(Token, Span)> {
        let offset = self.position_to_offset(content, position)?;
        let mut lexer = Lexer::new(content);

        while let Some((token, span)) = lexer.next_token() {
            if span.start <= offset && offset < span.end {
                return Some((token, span));
            }
            if span.start > offset {
                break;
            }
        }
        None
    }

    /// Convert a position to a byte offset.
    fn position_to_offset(&self, content: &str, position: Position) -> Option<usize> {
        let mut current_line = 0u32;
        let mut current_col = 0u32;

        for (i, ch) in content.char_indices() {
            if current_line == position.line && current_col == position.character {
                return Some(i);
            }
            if ch == '\n' {
                if current_line == position.line {
                    return Some(i);
                }
                current_line += 1;
                current_col = 0;
            } else {
                current_col += 1;
            }
        }

        if current_line == position.line {
            Some(content.len())
        } else {
            None
        }
    }

    /// Get hover information for a token.
    fn get_hover_info(&self, token: &Token) -> Option<String> {
        match token {
            // Morphemes
            Token::Tau => Some("**τ (tau)** - Transform/Map morpheme\n\nApplies a transformation to each element.\n\n```sigil\ndata|τ{_ * 2}  // Double each element\n```".to_string()),
            Token::Phi => Some("**φ (phi)** - Filter morpheme\n\nSelects elements matching a predicate.\n\n```sigil\ndata|φ{_ > 0}  // Keep positive values\n```".to_string()),
            Token::Sigma => Some("**σ (sigma)** - Sort morpheme\n\nOrders elements.\n\n```sigil\ndata|σ       // Sort ascending\ndata|σ.name  // Sort by field\n```".to_string()),
            Token::Rho => Some("**ρ (rho)** - Reduce morpheme\n\nFolds elements to a single value.\n\n```sigil\ndata|ρ+      // Sum\ndata|ρ*      // Product\n```".to_string()),
            Token::Lambda => Some("**λ (lambda)** - Anonymous function\n\n```sigil\nlet f = λ x -> x * 2\n```".to_string()),

            // Evidentiality
            Token::Bang => Some("**!** - Known evidentiality\n\nMarks a value as directly computed/verified.\n\n```sigil\nlet result! = compute()  // Known with certainty\n```".to_string()),
            Token::Question => Some("**?** - Uncertain evidentiality\n\nMarks a value that may be absent.\n\n```sigil\nlet maybe? = lookup(key)  // Might not exist\n```".to_string()),
            Token::Tilde => Some("**~** - Reported evidentiality\n\nMarks a value from an external source (untrusted).\n\n```sigil\nlet data~ = api.fetch()  // External source\n```".to_string()),
            Token::Interrobang => Some("**‽** - Paradox evidentiality\n\nMarks a trust boundary crossing.\n\n```sigil\nlet trusted‽ = unsafe { ptr }  // Trust assertion\n```".to_string()),

            // Keywords
            Token::Fn => Some("**fn** - Function declaration\n\n```sigil\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n```".to_string()),
            Token::Let => Some("**let** - Variable binding\n\n```sigil\nlet x = 42\nlet mut y = 0  // Mutable\n```".to_string()),
            Token::If => Some("**if** - Conditional expression\n\n```sigil\nif condition { then_branch } else { else_branch }\n```".to_string()),
            Token::Match => Some("**match** - Pattern matching\n\n```sigil\nmatch value {\n    Pattern => result,\n    _ => default,\n}\n```".to_string()),
            Token::Async => Some("**async** - Asynchronous function\n\n```sigil\nasync fn fetch() -> Data~ {\n    http.get(url)|await\n}\n```".to_string()),
            Token::Await => Some("**await** - Await async operation\n\n```sigil\nlet result = async_op()|await\n```".to_string()),
            Token::Actor => Some("**actor** - Actor definition\n\nDefines a concurrent actor with message handlers.\n\n```sigil\nactor Counter {\n    state: i64 = 0\n    on Increment(n) { self.state += n }\n}\n```".to_string()),
            Token::Struct => Some("**struct** - Structure definition\n\n```sigil\nstruct Point {\n    x: f64,\n    y: f64,\n}\n```".to_string()),
            Token::Enum => Some("**enum** - Enumeration definition\n\n```sigil\nenum Option<T> {\n    Some(T),\n    None,\n}\n```".to_string()),
            Token::Trait => Some("**trait** - Trait definition\n\n```sigil\ntrait Display {\n    fn fmt(self) -> str\n}\n```".to_string()),
            Token::Impl => Some("**impl** - Implementation block\n\n```sigil\nimpl Display for Point {\n    fn fmt(self) -> str { ... }\n}\n```".to_string()),

            // Operators
            Token::Pipe => Some("**|** - Pipe operator\n\nPasses a value through transformations.\n\n```sigil\ndata|filter|map|collect\n```".to_string()),
            Token::MiddleDot => Some("**·** - Incorporation operator\n\nFuses nouns and verbs into compound operations.\n\n```sigil\nfile·open·read·parse(path)\n```".to_string()),

            _ => None,
        }
    }

    /// Get completion items for a position.
    fn get_completions(&self, uri: &Url, _content: &str, _position: Position) -> Vec<CompletionItem> {
        let mut items = Vec::new();

        // Add symbols from the current document
        if let Some(doc) = self.documents.get(uri) {
            for sym in &doc.symbols {
                let (kind, detail) = match sym.kind {
                    SymbolKind::FUNCTION => (CompletionItemKind::FUNCTION, sym.detail.clone()),
                    SymbolKind::STRUCT => (CompletionItemKind::STRUCT, sym.detail.clone()),
                    SymbolKind::ENUM => (CompletionItemKind::ENUM, sym.detail.clone()),
                    SymbolKind::INTERFACE => (CompletionItemKind::INTERFACE, Some("trait".to_string())),
                    SymbolKind::VARIABLE => (CompletionItemKind::VARIABLE, Some("variable".to_string())),
                    SymbolKind::CONSTANT => (CompletionItemKind::CONSTANT, Some("constant".to_string())),
                    SymbolKind::CLASS => (CompletionItemKind::CLASS, Some("actor".to_string())),
                    _ => (CompletionItemKind::TEXT, None),
                };
                items.push(CompletionItem {
                    label: sym.name.clone(),
                    kind: Some(kind),
                    detail,
                    sort_text: Some(format!("0_{}", sym.name)), // Sort user symbols first
                    ..Default::default()
                });
            }
        }

        // Keywords
        for (label, detail, kind) in [
            ("fn", "Function declaration", CompletionItemKind::KEYWORD),
            ("let", "Variable binding", CompletionItemKind::KEYWORD),
            ("mut", "Mutable modifier", CompletionItemKind::KEYWORD),
            ("if", "Conditional", CompletionItemKind::KEYWORD),
            ("else", "Else branch", CompletionItemKind::KEYWORD),
            ("match", "Pattern matching", CompletionItemKind::KEYWORD),
            ("for", "For loop", CompletionItemKind::KEYWORD),
            ("while", "While loop", CompletionItemKind::KEYWORD),
            ("return", "Return value", CompletionItemKind::KEYWORD),
            ("async", "Async function", CompletionItemKind::KEYWORD),
            ("await", "Await async", CompletionItemKind::KEYWORD),
            ("struct", "Structure", CompletionItemKind::KEYWORD),
            ("enum", "Enumeration", CompletionItemKind::KEYWORD),
            ("trait", "Trait", CompletionItemKind::KEYWORD),
            ("impl", "Implementation", CompletionItemKind::KEYWORD),
            ("actor", "Actor definition", CompletionItemKind::KEYWORD),
            ("pub", "Public visibility", CompletionItemKind::KEYWORD),
            ("use", "Import", CompletionItemKind::KEYWORD),
        ] {
            items.push(CompletionItem {
                label: label.to_string(),
                kind: Some(kind),
                detail: Some(detail.to_string()),
                ..Default::default()
            });
        }

        // Morphemes with snippets
        for (label, insert, detail) in [
            ("τ", "τ{$1}", "Transform/map - τ{expr}"),
            ("φ", "φ{$1}", "Filter - φ{predicate}"),
            ("σ", "σ", "Sort"),
            ("ρ", "ρ$1", "Reduce - ρ+ or ρ{acc, x => ...}"),
            ("λ", "λ $1 -> $2", "Lambda - λ x -> expr"),
            ("Σ", "Σ", "Sum"),
            ("Π", "Π", "Product"),
            ("map", "|τ{$1}", "Transform (ASCII)"),
            ("filter", "|φ{$1}", "Filter (ASCII)"),
            ("sort", "|σ", "Sort (ASCII)"),
            ("reduce", "|ρ$1", "Reduce (ASCII)"),
            ("sum", "|Σ", "Sum (ASCII)"),
        ] {
            items.push(CompletionItem {
                label: label.to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some(detail.to_string()),
                insert_text: Some(insert.to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                ..Default::default()
            });
        }

        // Evidentiality markers
        for (label, detail) in [
            ("!", "Known - directly computed"),
            ("?", "Uncertain - may be absent"),
            ("~", "Reported - external source"),
            ("‽", "Paradox - trust boundary"),
        ] {
            items.push(CompletionItem {
                label: label.to_string(),
                kind: Some(CompletionItemKind::OPERATOR),
                detail: Some(detail.to_string()),
                ..Default::default()
            });
        }

        // Standard library functions
        for (label, detail) in [
            ("print", "Print to stdout"),
            ("println", "Print with newline"),
            ("len", "Get length"),
            ("push", "Push to array"),
            ("pop", "Pop from array"),
            ("sqrt", "Square root"),
            ("sin", "Sine"),
            ("cos", "Cosine"),
            ("abs", "Absolute value"),
            ("now", "Current timestamp"),
            ("known", "Mark as known evidentiality"),
            ("uncertain", "Mark as uncertain"),
            ("reported", "Mark as reported"),
            ("validate", "Validate and promote trust"),
        ] {
            items.push(CompletionItem {
                label: label.to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some(detail.to_string()),
                ..Default::default()
            });
        }

        // Types
        for (label, detail) in [
            ("i32", "32-bit signed integer"),
            ("i64", "64-bit signed integer"),
            ("u32", "32-bit unsigned integer"),
            ("u64", "64-bit unsigned integer"),
            ("f32", "32-bit float"),
            ("f64", "64-bit float"),
            ("bool", "Boolean"),
            ("str", "String slice"),
            ("String", "Owned string"),
            ("Vec", "Dynamic array"),
            ("Option", "Optional value"),
            ("Result", "Result type"),
        ] {
            items.push(CompletionItem {
                label: label.to_string(),
                kind: Some(CompletionItemKind::CLASS),
                detail: Some(detail.to_string()),
                ..Default::default()
            });
        }

        items
    }

    /// Extract symbols and type info from a document.
    fn extract_symbols(&self, content: &str) -> (Vec<SymbolDef>, HashMap<String, Vec<Span>>, HashMap<String, String>) {
        let mut symbols = Vec::new();
        let mut references: HashMap<String, Vec<Span>> = HashMap::new();
        let mut types: HashMap<String, String> = HashMap::new();

        // First, try to parse the AST to extract detailed type info
        let mut parser = Parser::new(content);
        if let Ok(ast) = parser.parse_file() {
            for spanned_item in &ast.items {
                match &spanned_item.node {
                    sigil_parser::Item::Function(func) => {
                        let sig = self.format_ast_function_signature(func);
                        types.insert(func.name.name.clone(), sig);
                    }
                    sigil_parser::Item::Struct(s) => {
                        let struct_repr = match &s.fields {
                            sigil_parser::ast::StructFields::Named(fields) => {
                                let field_strs: Vec<String> = fields.iter()
                                    .map(|f| format!("{}: {}", f.name.name, self.format_type(&f.ty)))
                                    .collect();
                                format!("struct {} {{ {} }}", s.name.name, field_strs.join(", "))
                            }
                            sigil_parser::ast::StructFields::Tuple(tys) => {
                                let ty_strs: Vec<String> = tys.iter()
                                    .map(|t| self.format_type(t))
                                    .collect();
                                format!("struct {}({})", s.name.name, ty_strs.join(", "))
                            }
                            sigil_parser::ast::StructFields::Unit => {
                                format!("struct {}", s.name.name)
                            }
                        };
                        types.insert(s.name.name.clone(), struct_repr);
                    }
                    sigil_parser::Item::Enum(e) => {
                        let variants: Vec<String> = e.variants.iter()
                            .map(|v| v.name.name.clone())
                            .collect();
                        types.insert(e.name.name.clone(), format!("enum {{ {} }}", variants.join(" | ")));
                    }
                    sigil_parser::Item::Trait(t) => {
                        types.insert(t.name.name.clone(), format!("trait {}", t.name.name));
                    }
                    _ => {}
                }
            }
        }

        // Then do lexer-based symbol extraction for locations and references
        let mut lexer = Lexer::new(content);
        let mut prev_token: Option<Token> = None;

        while let Some((token, span)) = lexer.next_token() {
            match (&prev_token, &token) {
                // Function definitions: fn name
                (Some(Token::Fn), Token::Ident(name)) => {
                    symbols.push(SymbolDef {
                        name: name.clone(),
                        kind: SymbolKind::FUNCTION,
                        span,
                        detail: types.get(name).cloned().or(Some("function".to_string())),
                    });
                }
                // Struct definitions: struct Name
                (Some(Token::Struct), Token::Ident(name)) => {
                    symbols.push(SymbolDef {
                        name: name.clone(),
                        kind: SymbolKind::STRUCT,
                        span,
                        detail: types.get(name).cloned().or(Some("struct".to_string())),
                    });
                }
                // Enum definitions: enum Name
                (Some(Token::Enum), Token::Ident(name)) => {
                    symbols.push(SymbolDef {
                        name: name.clone(),
                        kind: SymbolKind::ENUM,
                        span,
                        detail: types.get(name).cloned().or(Some("enum".to_string())),
                    });
                }
                // Trait definitions: trait Name
                (Some(Token::Trait), Token::Ident(name)) => {
                    symbols.push(SymbolDef {
                        name: name.clone(),
                        kind: SymbolKind::INTERFACE,
                        span,
                        detail: types.get(name).cloned().or(Some("trait".to_string())),
                    });
                }
                // Variable definitions: let name or let mut name
                (Some(Token::Let | Token::Mut), Token::Ident(name)) => {
                    symbols.push(SymbolDef {
                        name: name.clone(),
                        kind: SymbolKind::VARIABLE,
                        span,
                        detail: Some("variable".to_string()),
                    });
                }
                // Const definitions: const NAME
                (Some(Token::Const), Token::Ident(name)) => {
                    symbols.push(SymbolDef {
                        name: name.clone(),
                        kind: SymbolKind::CONSTANT,
                        span,
                        detail: Some("constant".to_string()),
                    });
                }
                // Actor definitions: actor Name
                (Some(Token::Actor), Token::Ident(name)) => {
                    symbols.push(SymbolDef {
                        name: name.clone(),
                        kind: SymbolKind::CLASS,
                        span,
                        detail: Some("actor".to_string()),
                    });
                }
                // Track all identifier usages as references
                (_, Token::Ident(name)) => {
                    references.entry(name.clone()).or_default().push(span);
                }
                _ => {}
            }
            prev_token = Some(token);
        }

        (symbols, references, types)
    }

    /// Format an AST function signature for display.
    fn format_ast_function_signature(&self, func: &sigil_parser::ast::Function) -> String {
        let params: Vec<String> = func.params.iter()
            .map(|p| {
                let name = self.format_pattern(&p.pattern);
                format!("{}: {}", name, self.format_type(&p.ty))
            })
            .collect();

        let ret = func.return_type.as_ref()
            .map(|t| format!(" -> {}", self.format_type(t)))
            .unwrap_or_default();

        format!("fn {}({}){}", func.name.name, params.join(", "), ret)
    }

    /// Format a pattern for display (used for parameter names).
    fn format_pattern(&self, pattern: &sigil_parser::ast::Pattern) -> String {
        match pattern {
            sigil_parser::ast::Pattern::Ident { name, .. } => name.name.clone(),
            sigil_parser::ast::Pattern::Wildcard => "_".to_string(),
            sigil_parser::ast::Pattern::Tuple(patterns) => {
                let ps: Vec<String> = patterns.iter().map(|p| self.format_pattern(p)).collect();
                format!("({})", ps.join(", "))
            }
            sigil_parser::ast::Pattern::Ref { mutable, pattern } => {
                if *mutable {
                    format!("&mut {}", self.format_pattern(pattern))
                } else {
                    format!("&{}", self.format_pattern(pattern))
                }
            }
            _ => "_".to_string(),
        }
    }

    /// Format a type expression for display.
    fn format_type(&self, ty: &sigil_parser::TypeExpr) -> String {
        match ty {
            sigil_parser::TypeExpr::Path(path) => self.format_type_path(path),
            sigil_parser::TypeExpr::Reference { lifetime, mutable, inner } => {
                let lt = lifetime.as_ref().map(|l| format!("'{} ", l)).unwrap_or_default();
                if *mutable {
                    format!("&{}mut {}", lt, self.format_type(inner))
                } else {
                    format!("&{}{}", lt, self.format_type(inner))
                }
            }
            sigil_parser::TypeExpr::Pointer { mutable, inner } => {
                if *mutable {
                    format!("*mut {}", self.format_type(inner))
                } else {
                    format!("*const {}", self.format_type(inner))
                }
            }
            sigil_parser::TypeExpr::Array { element, .. } => {
                format!("[{}; _]", self.format_type(element))
            }
            sigil_parser::TypeExpr::Slice(inner) => {
                format!("[{}]", self.format_type(inner))
            }
            sigil_parser::TypeExpr::Tuple(elements) => {
                let els: Vec<String> = elements.iter().map(|e| self.format_type(e)).collect();
                format!("({})", els.join(", "))
            }
            sigil_parser::TypeExpr::Function { params, return_type } => {
                let ps: Vec<String> = params.iter().map(|p| self.format_type(p)).collect();
                let ret = return_type.as_ref()
                    .map(|t| format!(" -> {}", self.format_type(t)))
                    .unwrap_or_default();
                format!("fn({}){}", ps.join(", "), ret)
            }
            sigil_parser::TypeExpr::Evidential { inner, evidentiality, error_type } => {
                let marker = match evidentiality {
                    sigil_parser::ast::Evidentiality::Known => "!",
                    sigil_parser::ast::Evidentiality::Uncertain => "?",
                    sigil_parser::ast::Evidentiality::Reported => "~",
                    sigil_parser::ast::Evidentiality::Predicted => "◊",
                    sigil_parser::ast::Evidentiality::Paradox => "‽",
                };
                let base = format!("{}{}", self.format_type(inner), marker);
                if let Some(err) = error_type {
                    format!("{}[{}]", base, self.format_type(err))
                } else {
                    base
                }
            }
            sigil_parser::TypeExpr::Cycle { .. } => "Cycle<_>".to_string(),
            sigil_parser::TypeExpr::Simd { element, lanes } => {
                format!("simd<{}, {}>", self.format_type(element), lanes)
            }
            sigil_parser::TypeExpr::Atomic(inner) => {
                format!("atomic<{}>", self.format_type(inner))
            }
            sigil_parser::TypeExpr::Lifetime(lt) => format!("'{}", lt),
            sigil_parser::TypeExpr::TraitObject(bounds) => {
                let bound_strs: Vec<String> = bounds.iter().map(|b| self.format_type(b)).collect();
                format!("dyn {}", bound_strs.join(" + "))
            }
            sigil_parser::TypeExpr::Hrtb { lifetimes, bound } => {
                format!("for<{}> {}", lifetimes.iter().map(|l| format!("'{}", l)).collect::<Vec<_>>().join(", "), self.format_type(bound))
            }
            sigil_parser::TypeExpr::InlineStruct { fields } => {
                let field_strs: Vec<String> = fields.iter()
                    .map(|f| format!("{}: {}", f.name.name, self.format_type(&f.ty)))
                    .collect();
                format!("struct {{ {} }}", field_strs.join(", "))
            }
            sigil_parser::TypeExpr::InlineEnum { variants } => {
                let var_strs: Vec<String> = variants.iter().map(|v| v.name.name.clone()).collect();
                format!("enum {{ {} }}", var_strs.join(" | "))
            }
            sigil_parser::TypeExpr::ImplTrait(bounds) => {
                let bound_strs: Vec<String> = bounds.iter().map(|b| self.format_type(b)).collect();
                format!("impl {}", bound_strs.join(" + "))
            }
            sigil_parser::TypeExpr::AssocTypeBinding { name, ty } => {
                format!("{} = {}", name.name, self.format_type(ty))
            }
            sigil_parser::TypeExpr::ConstExpr(_) => "<const>".to_string(),
            sigil_parser::TypeExpr::QualifiedPath { self_type, trait_path, item_path } => {
                let base = self.format_type(self_type);
                let item = self.format_type_path(item_path);
                if let Some(trait_p) = trait_path {
                    format!("<{} as {}>::{}", base, self.format_type_path(trait_p), item)
                } else {
                    format!("<{}>::{}", base, item)
                }
            }
            sigil_parser::TypeExpr::Infer => "_".to_string(),
            sigil_parser::TypeExpr::Never => "!".to_string(),
            sigil_parser::TypeExpr::Linear(inner) => format!("linear {}", self.format_type(inner)),
        }
    }

    /// Format a type path for display.
    fn format_type_path(&self, path: &sigil_parser::TypePath) -> String {
        path.segments.iter()
            .map(|seg| {
                if let Some(ref generics) = seg.generics {
                    let args: Vec<String> = generics.iter().map(|g| self.format_type(g)).collect();
                    format!("{}<{}>", seg.ident.name, args.join(", "))
                } else {
                    seg.ident.name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join("::")
    }

    /// Generate semantic tokens for a document.
    fn generate_semantic_tokens(&self, content: &str) -> Vec<SemanticToken> {
        let mut tokens = Vec::new();
        let mut lexer = Lexer::new(content);
        let mut prev_line = 0u32;
        let mut prev_char = 0u32;

        while let Some((token, span)) = lexer.next_token() {
            let token_type = match &token {
                // Keywords
                Token::Fn | Token::Let | Token::Mut | Token::Const | Token::If |
                Token::Else | Token::Match | Token::For | Token::While | Token::Loop |
                Token::Return | Token::Break | Token::Continue | Token::Struct |
                Token::Enum | Token::Trait | Token::Impl | Token::Pub | Token::Use |
                Token::Mod | Token::Async | Token::Await | Token::Actor | Token::Type |
                Token::Where | Token::In | Token::As | Token::Static => Some(SemanticTokenType::Keyword),

                // Morphemes
                Token::Tau | Token::Phi | Token::Sigma | Token::Rho |
                Token::Lambda | Token::Pi | Token::Hourglass => Some(SemanticTokenType::Morpheme),

                // Evidentiality markers
                Token::Bang | Token::Question | Token::Tilde |
                Token::Interrobang => Some(SemanticTokenType::Evidentiality),

                // Operators
                Token::Pipe | Token::MiddleDot | Token::Arrow | Token::FatArrow |
                Token::Plus | Token::Minus | Token::Star | Token::Slash |
                Token::EqEq | Token::NotEq | Token::Lt | Token::Gt |
                Token::AndAnd | Token::OrOr => Some(SemanticTokenType::Operator),

                // Comments
                Token::LineComment(_) | Token::DocComment(_) => Some(SemanticTokenType::Comment),

                // Strings
                Token::StringLit(_) | Token::CharLit(_) => Some(SemanticTokenType::String),

                // Numbers
                Token::IntLit(_) | Token::FloatLit(_) | Token::BinaryLit(_) |
                Token::HexLit(_) | Token::OctalLit(_) => Some(SemanticTokenType::Number),

                // Identifiers (would need context to distinguish function vs variable)
                Token::Ident(_) => Some(SemanticTokenType::Variable),

                _ => None,
            };

            if let Some(tt) = token_type {
                let (line, col) = self.offset_to_line_col(content, span.start);
                let length = (span.end - span.start) as u32;

                // Calculate delta from previous token
                let delta_line = line - prev_line;
                let delta_start = if delta_line == 0 {
                    col - prev_char
                } else {
                    col
                };

                tokens.push(SemanticToken {
                    delta_line,
                    delta_start,
                    length,
                    token_type: tt.to_index(),
                    token_modifiers_bitset: 0,
                });

                prev_line = line;
                prev_char = col;
            }
        }

        tokens
    }

    /// Convert byte offset to line/column.
    fn offset_to_line_col(&self, content: &str, offset: usize) -> (u32, u32) {
        let mut line = 0u32;
        let mut col = 0u32;

        for (i, ch) in content.char_indices() {
            if i >= offset {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }

        (line, col)
    }

    /// Find the definition of a symbol at the given position.
    fn find_definition(&self, uri: &Url, position: Position) -> Option<Location> {
        let doc = self.documents.get(uri)?;
        let content = doc.content.to_string();
        let offset = self.position_to_offset(&content, position)?;

        // Get the token at position
        let mut lexer = Lexer::new(&content);
        let mut target_name: Option<String> = None;

        while let Some((token, span)) = lexer.next_token() {
            if span.start <= offset && offset < span.end {
                if let Token::Ident(name) = token {
                    target_name = Some(name);
                }
                break;
            }
        }

        let name = target_name?;

        // Search for definition in symbols
        for sym in &doc.symbols {
            if sym.name == name {
                let (line, col, end_col) = self.span_to_position(&content, Some(sym.span));
                return Some(Location {
                    uri: uri.clone(),
                    range: Range {
                        start: Position { line, character: col },
                        end: Position { line, character: end_col },
                    },
                });
            }
        }

        None
    }
}

// Helper trait to get span from parse errors
trait ParseErrorExt {
    fn span(&self) -> Option<Span>;
}

impl ParseErrorExt for sigil_parser::parser::ParseError {
    fn span(&self) -> Option<Span> {
        match self {
            sigil_parser::parser::ParseError::UnexpectedToken { span, .. } => Some(*span),
            _ => None,
        }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for OracleServer {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        info!("Oracle LSP server initializing");

        // Build semantic tokens legend
        let token_types: Vec<tower_lsp::lsp_types::SemanticTokenType> = SemanticTokenType::legend()
            .into_iter()
            .map(|t| match t {
                SemanticTokenType::Keyword => tower_lsp::lsp_types::SemanticTokenType::KEYWORD,
                SemanticTokenType::Function => tower_lsp::lsp_types::SemanticTokenType::FUNCTION,
                SemanticTokenType::Variable => tower_lsp::lsp_types::SemanticTokenType::VARIABLE,
                SemanticTokenType::Parameter => tower_lsp::lsp_types::SemanticTokenType::PARAMETER,
                SemanticTokenType::Type => tower_lsp::lsp_types::SemanticTokenType::TYPE,
                SemanticTokenType::Morpheme => tower_lsp::lsp_types::SemanticTokenType::MACRO,
                SemanticTokenType::Evidentiality => tower_lsp::lsp_types::SemanticTokenType::MODIFIER,
                SemanticTokenType::Operator => tower_lsp::lsp_types::SemanticTokenType::OPERATOR,
                SemanticTokenType::Comment => tower_lsp::lsp_types::SemanticTokenType::COMMENT,
                SemanticTokenType::String => tower_lsp::lsp_types::SemanticTokenType::STRING,
                SemanticTokenType::Number => tower_lsp::lsp_types::SemanticTokenType::NUMBER,
            })
            .collect();

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![
                        "|".to_string(),
                        ".".to_string(),
                        ":".to_string(),
                        "·".to_string(),
                    ]),
                    ..Default::default()
                }),
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                    retrigger_characters: Some(vec![",".to_string()]),
                    ..Default::default()
                }),
                definition_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(
                        SemanticTokensOptions {
                            legend: SemanticTokensLegend {
                                token_types,
                                token_modifiers: vec![],
                            },
                            full: Some(SemanticTokensFullOptions::Bool(true)),
                            range: None,
                            ..Default::default()
                        },
                    ),
                ),
                diagnostic_provider: Some(DiagnosticServerCapabilities::Options(
                    DiagnosticOptions {
                        inter_file_dependencies: false,
                        workspace_diagnostics: false,
                        ..Default::default()
                    },
                )),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "sigil-oracle".to_string(),
                version: Some("0.1.0".to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        info!("Oracle LSP server initialized");
        self.client
            .log_message(MessageType::INFO, "Sigil Oracle LSP ready")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Oracle LSP server shutting down");
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let content = params.text_document.text;
        let version = params.text_document.version;

        info!("Document opened: {}", uri);

        // Extract symbols and type info
        let (symbols, references, types) = self.extract_symbols(&content);

        self.documents.insert(
            uri.clone(),
            Document {
                content: Rope::from_str(&content),
                version,
                symbols,
                references,
                types,
            },
        );

        // Analyze and publish diagnostics
        let diagnostics = self.analyze_document(&uri).await;
        self.client
            .publish_diagnostics(uri, diagnostics, Some(version))
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;

        if let Some(change) = params.content_changes.into_iter().last() {
            // Extract symbols and type info from new content
            let (symbols, references, types) = self.extract_symbols(&change.text);

            if let Some(mut doc) = self.documents.get_mut(&uri) {
                doc.content = Rope::from_str(&change.text);
                doc.version = version;
                doc.symbols = symbols;
                doc.references = references;
                doc.types = types;
            }
        }

        // Re-analyze and publish diagnostics
        let diagnostics = self.analyze_document(&uri).await;
        self.client
            .publish_diagnostics(uri, diagnostics, Some(version))
            .await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        info!("Document closed: {}", uri);
        self.documents.remove(&uri);

        // Clear diagnostics
        self.client
            .publish_diagnostics(uri, vec![], None)
            .await;
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };

        let content = doc.content.to_string();

        if let Some((token, _span)) = self.get_token_at_position(&content, position) {
            // First check for built-in keyword/morpheme hover info
            if let Some(info) = self.get_hover_info(&token) {
                return Ok(Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: info,
                    }),
                    range: None,
                }));
            }

            // Check for user-defined symbol type info
            if let Token::Ident(name) = &token {
                if let Some(type_info) = doc.types.get(name) {
                    return Ok(Some(Hover {
                        contents: HoverContents::Markup(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: format!("```sigil\n{}\n```", type_info),
                        }),
                        range: None,
                    }));
                }
            }
        }

        Ok(None)
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };

        let content = doc.content.to_string();
        let offset = match self.position_to_offset(&content, position) {
            Some(o) => o,
            None => return Ok(None),
        };

        // Search backwards for an opening paren to find the function name
        let before = &content[..offset];
        let mut paren_depth = 0;
        let mut func_name_end = None;

        for (i, c) in before.char_indices().rev() {
            match c {
                ')' => paren_depth += 1,
                '(' => {
                    if paren_depth == 0 {
                        func_name_end = Some(i);
                        break;
                    }
                    paren_depth -= 1;
                }
                _ => {}
            }
        }

        let func_name_end = match func_name_end {
            Some(e) => e,
            None => return Ok(None),
        };

        // Extract the function name (scan backwards for identifier)
        // Support Unicode identifiers (letters, numbers, underscore, and Unicode XID characters)
        let before_paren = &content[..func_name_end];
        let func_name: String = before_paren
            .chars()
            .rev()
            .take_while(|c| c.is_alphanumeric() || *c == '_' || is_xid_continue(*c))
            .collect::<String>()
            .chars()
            .rev()
            .collect();

        if func_name.is_empty() {
            return Ok(None);
        }

        // Look up the function signature
        if let Some(sig) = doc.types.get(&func_name) {
            // Count commas to determine active parameter
            let inside_parens = &content[func_name_end + 1..offset];
            let mut active_param = 0u32;
            let mut depth = 0;
            for c in inside_parens.chars() {
                match c {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => depth -= 1,
                    ',' if depth == 0 => active_param += 1,
                    _ => {}
                }
            }

            return Ok(Some(SignatureHelp {
                signatures: vec![SignatureInformation {
                    label: sig.clone(),
                    documentation: None,
                    parameters: None, // Could extract parameters for highlighting
                    active_parameter: Some(active_param),
                }],
                active_signature: Some(0),
                active_parameter: Some(active_param),
            }));
        }

        Ok(None)
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let content = match self.documents.get(uri) {
            Some(doc) => doc.content.to_string(),
            None => return Ok(None),
        };

        let items = self.get_completions(uri, &content, position);
        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        if let Some(location) = self.find_definition(uri, position) {
            Ok(Some(GotoDefinitionResponse::Scalar(location)))
        } else {
            Ok(None)
        }
    }

    async fn references(
        &self,
        params: ReferenceParams,
    ) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };

        let content = doc.content.to_string();
        let offset = match self.position_to_offset(&content, position) {
            Some(o) => o,
            None => return Ok(None),
        };

        // Find the identifier at the cursor position
        let mut lexer = Lexer::new(&content);
        let mut target_name: Option<String> = None;

        while let Some((token, span)) = lexer.next_token() {
            if span.start <= offset && offset < span.end {
                if let Token::Ident(name) = token {
                    target_name = Some(name);
                }
                break;
            }
        }

        let name = match target_name {
            Some(n) => n,
            None => return Ok(None),
        };

        // Get all references to this name
        let refs = match doc.references.get(&name) {
            Some(r) => r,
            None => return Ok(None),
        };

        let locations: Vec<Location> = refs
            .iter()
            .map(|span| {
                let (line, col, end_col) = self.span_to_position(&content, Some(*span));
                Location {
                    uri: uri.clone(),
                    range: Range {
                        start: Position { line, character: col },
                        end: Position { line, character: end_col },
                    },
                }
            })
            .collect();

        if locations.is_empty() {
            Ok(None)
        } else {
            Ok(Some(locations))
        }
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };

        let content = doc.content.to_string();
        let symbols: Vec<SymbolInformation> = doc
            .symbols
            .iter()
            .map(|sym| {
                let (line, col, end_col) = self.span_to_position(&content, Some(sym.span));
                #[allow(deprecated)]
                SymbolInformation {
                    name: sym.name.clone(),
                    kind: sym.kind,
                    tags: None,
                    deprecated: None,
                    location: Location {
                        uri: uri.clone(),
                        range: Range {
                            start: Position { line, character: col },
                            end: Position { line, character: end_col },
                        },
                    },
                    container_name: None,
                }
            })
            .collect();

        Ok(Some(DocumentSymbolResponse::Flat(symbols)))
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = &params.text_document.uri;

        let content = match self.documents.get(uri) {
            Some(doc) => doc.content.to_string(),
            None => return Ok(None),
        };

        let tokens = self.generate_semantic_tokens(&content);
        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data: tokens,
        })))
    }
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("sigil_oracle=info".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .init();

    info!("Starting Sigil Oracle LSP server");

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(OracleServer::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
