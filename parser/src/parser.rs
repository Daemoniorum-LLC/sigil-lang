//! Recursive descent parser for Sigil.
//!
//! Handles polysynthetic pipe chains, morpheme expressions, and evidentiality.

use crate::ast::*;
use crate::lexer::{Lexer, Token};
use crate::span::{Span, Spanned};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Unexpected token: expected {expected}, found {found:?} at {span}")]
    UnexpectedToken {
        expected: String,
        found: Token,
        span: Span,
    },
    #[error("Unexpected end of file")]
    UnexpectedEof,
    #[error("Invalid number literal: {0}")]
    InvalidNumber(String),
    #[error("Parse error: {0}")]
    Custom(String),
}

pub type ParseResult<T> = Result<T, ParseError>;

/// Recursive descent parser for Sigil.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Option<(Token, Span)>,
    /// Tracks whether we're parsing a condition (if/while/for) where < is comparison not generics
    in_condition: bool,
    /// Tracks if we have a pending `>` from splitting `>>` (Shr) in generic contexts
    pending_gt: Option<Span>,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let current = lexer.next_token();
        Self {
            lexer,
            current,
            in_condition: false,
            pending_gt: None,
        }
    }

    /// Parse a complete source file.
    pub fn parse_file(&mut self) -> ParseResult<SourceFile> {
        // Parse inner attributes first (#![...])
        let mut attrs = Vec::new();
        while matches!(self.current_token(), Some(Token::HashBang)) {
            attrs.push(self.parse_inner_attribute()?);
        }

        // Build crate config from attributes
        let config = self.build_crate_config(&attrs);

        let mut items = Vec::new();
        while !self.is_eof() {
            // Skip comments
            while matches!(
                self.current_token(),
                Some(
                    Token::LineComment(_)
                        | Token::TildeComment(_)
                        | Token::BlockComment(_)
                        | Token::DocComment(_)
                )
            ) {
                self.advance();
            }
            if self.is_eof() {
                break;
            }
            items.push(self.parse_item()?);
        }
        Ok(SourceFile {
            attrs,
            config,
            items,
        })
    }

    /// Parse an inner attribute: `#![name]` or `#![name(args)]`
    fn parse_inner_attribute(&mut self) -> ParseResult<Attribute> {
        self.expect(Token::HashBang)?;
        self.expect(Token::LBracket)?;

        let name = self.parse_ident()?;
        let args = self.parse_attr_args()?;

        self.expect(Token::RBracket)?;

        Ok(Attribute {
            name,
            args,
            is_inner: true,
        })
    }

    /// Parse an outer attribute: `#[name]` or `#[name(args)]` or `@[name]` or `@[name(args)]`
    /// The `@[]` syntax is the preferred Sigil style.
    /// Also supports the shorthand `@[Clone, Debug]` which expands to `@[derive(Clone, Debug)]`
    fn parse_outer_attribute(&mut self) -> ParseResult<Attribute> {
        // Accept either # or @ as attribute prefix
        let is_at_syntax = self.check(&Token::At);
        if is_at_syntax {
            self.advance();
        } else {
            self.expect(Token::Hash)?;
        }
        self.expect(Token::LBracket)?;

        let name = self.parse_attr_name()?;

        // Check for @[Ident, Ident, ...] shorthand for derive
        // Only applies to @ syntax when we see a comma after the first identifier
        if is_at_syntax && self.check(&Token::Comma) && !self.check(&Token::LParen) {
            // This is the shorthand @[Clone, Debug] syntax - convert to derive
            let mut args = vec![AttrArg::Ident(name)];
            while self.consume_if(&Token::Comma) {
                if self.check(&Token::RBracket) {
                    break; // Trailing comma
                }
                args.push(AttrArg::Ident(self.parse_attr_name()?));
            }
            self.expect(Token::RBracket)?;

            // Return as derive attribute
            Ok(Attribute {
                name: Ident {
                    name: "derive".to_string(),
                    evidentiality: None,
                    affect: None,
                    span: self.current_span(),
                },
                args: Some(AttrArgs::Paren(args)),
                is_inner: false,
            })
        } else {
            let args = self.parse_attr_args()?;
            self.expect(Token::RBracket)?;

            Ok(Attribute {
                name,
                args,
                is_inner: false,
            })
        }
    }

    /// Parse an attribute name (identifier, keyword, or path like async_trait::async_trait).
    fn parse_attr_name(&mut self) -> ParseResult<Ident> {
        let span = self.current_span();
        let first_name = match self.current_token().cloned() {
            Some(Token::Ident(name)) => {
                self.advance();
                name
            }
            // Handle keywords that can be used as attribute names
            Some(Token::Naked) => {
                self.advance();
                "naked".to_string()
            }
            Some(Token::Unsafe) => {
                self.advance();
                "unsafe".to_string()
            }
            Some(Token::Asm) => {
                self.advance();
                "asm".to_string()
            }
            Some(Token::Volatile) => {
                self.advance();
                "volatile".to_string()
            }
            Some(Token::Derive) => {
                self.advance();
                "derive".to_string()
            }
            Some(Token::Simd) => {
                self.advance();
                "simd".to_string()
            }
            Some(Token::Atomic) => {
                self.advance();
                "atomic".to_string()
            }
            Some(Token::Macro) => {
                self.advance();
                "macro".to_string()
            }
            Some(t) => {
                return Err(ParseError::UnexpectedToken {
                    expected: "attribute name".to_string(),
                    found: t,
                    span,
                })
            }
            None => return Err(ParseError::UnexpectedEof),
        };

        // Check for path continuation: attr_name::next_segment::...
        let mut full_name = first_name;
        while self.consume_if(&Token::ColonColon) {
            let segment = match self.current_token().cloned() {
                Some(Token::Ident(name)) => {
                    self.advance();
                    name
                }
                Some(t) => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "identifier after ::".to_string(),
                        found: t,
                        span: self.current_span(),
                    })
                }
                None => return Err(ParseError::UnexpectedEof),
            };
            full_name = format!("{}::{}", full_name, segment);
        }

        Ok(Ident {
            name: full_name,
            evidentiality: None,
            affect: None,
            span,
        })
    }

    /// Parse attribute arguments if present.
    fn parse_attr_args(&mut self) -> ParseResult<Option<AttrArgs>> {
        if self.consume_if(&Token::LParen) {
            let mut args = Vec::new();

            while !self.check(&Token::RParen) {
                args.push(self.parse_attr_arg()?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }

            self.expect(Token::RParen)?;
            Ok(Some(AttrArgs::Paren(args)))
        } else if self.consume_if(&Token::Eq) {
            let expr = self.parse_expr()?;
            Ok(Some(AttrArgs::Eq(Box::new(expr))))
        } else {
            Ok(None)
        }
    }

    /// Parse a single attribute argument.
    fn parse_attr_arg(&mut self) -> ParseResult<AttrArg> {
        match self.current_token().cloned() {
            Some(Token::StringLit(s)) => {
                self.advance();
                Ok(AttrArg::Literal(Literal::String(s)))
            }
            Some(Token::IntLit(s)) => {
                self.advance();
                Ok(AttrArg::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Decimal,
                    suffix: None,
                }))
            }
            Some(Token::HexLit(s)) => {
                self.advance();
                Ok(AttrArg::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Hex,
                    suffix: None,
                }))
            }
            Some(Token::BinaryLit(s)) => {
                self.advance();
                Ok(AttrArg::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Binary,
                    suffix: None,
                }))
            }
            Some(Token::OctalLit(s)) => {
                self.advance();
                Ok(AttrArg::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Octal,
                    suffix: None,
                }))
            }
            Some(Token::Ident(_)) => {
                let ident = self.parse_ident()?;
                self.parse_attr_arg_after_ident(ident)
            }
            // Handle keywords that might appear as feature names in attributes
            Some(Token::Asm) => {
                let span = self.current_span();
                self.advance();
                let ident = Ident {
                    name: "asm".to_string(),
                    evidentiality: None,
                    affect: None,
                    span,
                };
                self.parse_attr_arg_after_ident(ident)
            }
            Some(Token::Volatile) => {
                let span = self.current_span();
                self.advance();
                let ident = Ident {
                    name: "volatile".to_string(),
                    evidentiality: None,
                    affect: None,
                    span,
                };
                self.parse_attr_arg_after_ident(ident)
            }
            Some(Token::Naked) => {
                let span = self.current_span();
                self.advance();
                let ident = Ident {
                    name: "naked".to_string(),
                    evidentiality: None,
                    affect: None,
                    span,
                };
                self.parse_attr_arg_after_ident(ident)
            }
            Some(Token::Packed) => {
                let span = self.current_span();
                self.advance();
                let ident = Ident {
                    name: "packed".to_string(),
                    evidentiality: None,
                    affect: None,
                    span,
                };
                self.parse_attr_arg_after_ident(ident)
            }
            Some(Token::Unsafe) => {
                let span = self.current_span();
                self.advance();
                let ident = Ident {
                    name: "unsafe".to_string(),
                    evidentiality: None,
                    affect: None,
                    span,
                };
                self.parse_attr_arg_after_ident(ident)
            }
            Some(t) => Err(ParseError::UnexpectedToken {
                expected: "attribute argument".to_string(),
                found: t,
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    /// Helper to continue parsing after an identifier in an attribute argument.
    fn parse_attr_arg_after_ident(&mut self, ident: Ident) -> ParseResult<AttrArg> {
        // Check for qualified path like serde::Serialize
        if self.consume_if(&Token::ColonColon) || self.consume_if(&Token::MiddleDot) {
            let mut path_parts = vec![ident.name.clone()];
            loop {
                let part = self.parse_ident()?;
                path_parts.push(part.name);
                if !self.consume_if(&Token::ColonColon) && !self.consume_if(&Token::MiddleDot) {
                    break;
                }
            }
            // Return the full qualified path as an identifier with :: separators
            let full_path = path_parts.join("::");
            return Ok(AttrArg::Ident(Ident {
                name: full_path,
                evidentiality: None,
                affect: None,
                span: ident.span.clone(),
            }));
        }
        // Check for key = value
        if self.consume_if(&Token::Eq) {
            let value = self.parse_expr()?;
            Ok(AttrArg::KeyValue {
                key: ident,
                value: Box::new(value),
            })
        }
        // Check for nested attr(...)
        else if self.check(&Token::LParen) {
            let args = self.parse_attr_args()?;
            Ok(AttrArg::Nested(Attribute {
                name: ident,
                args,
                is_inner: false,
            }))
        }
        // Just an identifier
        else {
            Ok(AttrArg::Ident(ident))
        }
    }

    /// Parse interpolation parts from a string like "hello {name}, you are {age} years old"
    /// Returns a vector of text segments and expression segments.
    fn parse_interpolation_parts(&mut self, s: &str) -> ParseResult<Vec<InterpolationPart>> {
        let mut parts = Vec::new();
        let mut current_text = String::new();
        let mut chars = s.chars().peekable();
        let mut brace_depth = 0;
        let mut expr_content = String::new();
        let mut in_expr = false;

        while let Some(c) = chars.next() {
            if in_expr {
                if c == '{' {
                    brace_depth += 1;
                    expr_content.push(c);
                } else if c == '}' {
                    if brace_depth > 0 {
                        brace_depth -= 1;
                        expr_content.push(c);
                    } else {
                        // End of expression - parse it
                        in_expr = false;
                        if !expr_content.is_empty() {
                            // Parse the expression content
                            let mut expr_parser = Parser::new(&expr_content);
                            match expr_parser.parse_expr() {
                                Ok(expr) => {
                                    parts.push(InterpolationPart::Expr(Box::new(expr)));
                                }
                                Err(_) => {
                                    // If parsing fails, treat as text
                                    parts.push(InterpolationPart::Text(format!(
                                        "{{{}}}",
                                        expr_content
                                    )));
                                }
                            }
                        }
                        expr_content.clear();
                    }
                } else {
                    expr_content.push(c);
                }
            } else if c == '{' {
                if chars.peek() == Some(&'{') {
                    // Escaped brace {{
                    chars.next();
                    current_text.push('{');
                } else {
                    // Start of expression
                    if !current_text.is_empty() {
                        parts.push(InterpolationPart::Text(current_text.clone()));
                        current_text.clear();
                    }
                    in_expr = true;
                }
            } else if c == '}' {
                if chars.peek() == Some(&'}') {
                    // Escaped brace }}
                    chars.next();
                    current_text.push('}');
                } else {
                    current_text.push(c);
                }
            } else {
                current_text.push(c);
            }
        }

        // Add any remaining text
        if !current_text.is_empty() {
            parts.push(InterpolationPart::Text(current_text));
        }

        // If we have no parts, add an empty text part
        if parts.is_empty() {
            parts.push(InterpolationPart::Text(String::new()));
        }

        Ok(parts)
    }

    /// Build crate configuration from parsed inner attributes.
    fn build_crate_config(&self, attrs: &[Attribute]) -> CrateConfig {
        let mut config = CrateConfig::default();
        let mut linker = LinkerConfig::default();
        let mut has_linker_config = false;

        for attr in attrs {
            match attr.name.name.as_str() {
                "no_std" => config.no_std = true,
                "no_main" => config.no_main = true,
                "feature" => {
                    if let Some(AttrArgs::Paren(args)) = &attr.args {
                        for arg in args {
                            if let AttrArg::Ident(ident) = arg {
                                config.features.push(ident.name.clone());
                            }
                        }
                    }
                }
                "target" => {
                    let mut target = TargetConfig::default();
                    if let Some(AttrArgs::Paren(args)) = &attr.args {
                        for arg in args {
                            if let AttrArg::KeyValue { key, value } = arg {
                                if let Expr::Literal(Literal::String(s)) = value.as_ref() {
                                    match key.name.as_str() {
                                        "arch" => target.arch = Some(s.clone()),
                                        "os" => target.os = Some(s.clone()),
                                        "abi" => target.abi = Some(s.clone()),
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    config.target = Some(target);
                }
                // Linker configuration attributes
                "linker_script" => {
                    if let Some(AttrArgs::Eq(value)) = &attr.args {
                        if let Expr::Literal(Literal::String(s)) = value.as_ref() {
                            linker.script = Some(s.clone());
                            has_linker_config = true;
                        }
                    }
                }
                "entry_point" => {
                    if let Some(AttrArgs::Eq(value)) = &attr.args {
                        if let Expr::Literal(Literal::String(s)) = value.as_ref() {
                            linker.entry_point = Some(s.clone());
                            has_linker_config = true;
                        }
                    }
                }
                "base_address" => {
                    if let Some(AttrArgs::Eq(value)) = &attr.args {
                        if let Expr::Literal(Literal::Int { value: s, base, .. }) = value.as_ref() {
                            let addr = Self::parse_int_value(s, *base);
                            linker.base_address = Some(addr);
                            has_linker_config = true;
                        }
                    }
                }
                "stack_size" => {
                    if let Some(AttrArgs::Eq(value)) = &attr.args {
                        if let Expr::Literal(Literal::Int { value: s, base, .. }) = value.as_ref() {
                            let size = Self::parse_int_value(s, *base);
                            linker.stack_size = Some(size);
                            has_linker_config = true;
                        }
                    }
                }
                "link" => {
                    // #![link(flag = "-nostdlib", flag = "-static")]
                    if let Some(AttrArgs::Paren(args)) = &attr.args {
                        for arg in args {
                            if let AttrArg::KeyValue { key, value } = arg {
                                if key.name == "flag" {
                                    if let Expr::Literal(Literal::String(s)) = value.as_ref() {
                                        linker.flags.push(s.clone());
                                        has_linker_config = true;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if has_linker_config {
            config.linker = Some(linker);
        }

        config
    }

    /// Parse an integer value from string with given base.
    fn parse_int_value(s: &str, base: NumBase) -> u64 {
        // Strip prefix based on base
        let (stripped, radix) = match base {
            NumBase::Binary => (
                s.strip_prefix("0b").or(s.strip_prefix("0B")).unwrap_or(s),
                2,
            ),
            NumBase::Octal => (
                s.strip_prefix("0o").or(s.strip_prefix("0O")).unwrap_or(s),
                8,
            ),
            NumBase::Decimal => (s, 10),
            NumBase::Hex => (
                s.strip_prefix("0x").or(s.strip_prefix("0X")).unwrap_or(s),
                16,
            ),
            NumBase::Vigesimal => (
                s.strip_prefix("0v").or(s.strip_prefix("0V")).unwrap_or(s),
                20,
            ),
            NumBase::Duodecimal => (
                s.strip_prefix("0d").or(s.strip_prefix("0D")).unwrap_or(s),
                12,
            ),
            NumBase::Sexagesimal => (
                s.strip_prefix("0s").or(s.strip_prefix("0S")).unwrap_or(s),
                60,
            ),
            NumBase::Explicit(r) => (s, r as u32),
        };
        // Remove underscores (numeric separators) and parse
        let clean: String = stripped.chars().filter(|c| *c != '_').collect();
        u64::from_str_radix(&clean, radix).unwrap_or(0)
    }

    // === Token utilities ===

    pub(crate) fn current_token(&self) -> Option<&Token> {
        self.current.as_ref().map(|(t, _)| t)
    }

    pub(crate) fn current_span(&self) -> Span {
        self.current.as_ref().map(|(_, s)| *s).unwrap_or_default()
    }

    pub(crate) fn advance(&mut self) -> Option<(Token, Span)> {
        let prev = self.current.take();
        self.current = self.lexer.next_token();
        prev
    }

    pub(crate) fn is_eof(&self) -> bool {
        self.current.is_none()
    }

    pub(crate) fn expect(&mut self, expected: Token) -> ParseResult<Span> {
        match &self.current {
            Some((token, span))
                if std::mem::discriminant(token) == std::mem::discriminant(&expected) =>
            {
                let span = *span;
                self.advance();
                Ok(span)
            }
            Some((token, span)) => Err(ParseError::UnexpectedToken {
                expected: format!("{:?}", expected),
                found: token.clone(),
                span: *span,
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    pub(crate) fn check(&self, expected: &Token) -> bool {
        matches!(&self.current, Some((token, _)) if std::mem::discriminant(token) == std::mem::discriminant(expected))
    }

    /// Peek at the next token (after current) without consuming anything.
    pub(crate) fn peek_next(&mut self) -> Option<&Token> {
        self.lexer.peek().map(|(t, _)| t)
    }

    /// Peek n tokens ahead (0 = next token after current, 1 = token after that, etc.)
    pub(crate) fn peek_n(&mut self, n: usize) -> Option<&Token> {
        self.lexer.peek_n(n).map(|(t, _)| t)
    }

    pub(crate) fn consume_if(&mut self, expected: &Token) -> bool {
        if self.check(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Skip any comments
    pub(crate) fn skip_comments(&mut self) {
        while matches!(
            self.current_token(),
            Some(Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_))
                | Some(Token::DocComment(_))
        ) {
            self.advance();
        }
    }

    /// Check if the current token is `>`, including pending `>` from split `>>`.
    pub(crate) fn check_gt(&self) -> bool {
        self.pending_gt.is_some() || self.check(&Token::Gt)
    }

    /// Expect a `>` token, handling the case where `>>` (Shr) needs to be split.
    /// This is necessary for nested generics like `Vec<Option<T>>`.
    pub(crate) fn expect_gt(&mut self) -> ParseResult<Span> {
        // First check if we have a pending `>` from a previous split
        if let Some(span) = self.pending_gt.take() {
            return Ok(span);
        }

        match &self.current {
            Some((Token::Gt, span)) => {
                let span = *span;
                self.advance();
                Ok(span)
            }
            Some((Token::Shr, span)) => {
                // Split `>>` into two `>` tokens
                // Take the first `>` now and save the second for later
                let span = *span;
                self.pending_gt = Some(span);
                self.advance();
                Ok(span)
            }
            Some((token, span)) => Err(ParseError::UnexpectedToken {
                expected: "Gt".to_string(),
                found: token.clone(),
                span: *span,
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    /// Consume a `>` if present, handling pending `>` from split `>>`.
    fn consume_gt(&mut self) -> bool {
        if self.pending_gt.is_some() {
            self.pending_gt = None;
            return true;
        }
        if self.check(&Token::Gt) {
            self.advance();
            return true;
        }
        // Also handle `>>` - split it and return first `>`
        if self.check(&Token::Shr) {
            let span = self.current_span();
            self.pending_gt = Some(span);
            self.advance();
            return true;
        }
        false
    }

    /// Check if the current token can start a new item.
    /// Used to make semicolons optional in Sigil.
    fn can_start_item(&self) -> bool {
        matches!(
            self.current_token(),
            Some(
                Token::Pub
                    | Token::Fn
                    | Token::Async
                    | Token::Struct
                    | Token::Enum
                    | Token::Trait
                    | Token::Impl
                    | Token::Type
                    | Token::Mod
                    | Token::Use
                    | Token::Const
                    | Token::Static
                    | Token::Actor
                    | Token::Extern
                    | Token::Hash
                    | Token::At
                    | Token::Naked
                    | Token::Packed
                    | Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_)
                    | Token::DocComment(_)
                    // Plurality keywords
                    | Token::Alter
                    | Token::Headspace
                    | Token::Reality
                    | Token::CoCon
            )
        ) || (matches!(self.current_token(), Some(Token::On)) && self.peek_next_is_trigger())
    }

    /// Check if peek is the Trigger token (for "on trigger" detection)
    fn peek_next_is_trigger(&self) -> bool {
        // This is a workaround since we can't mutably borrow in can_start_item
        false // Will be true when "on trigger" is seen
    }

    /// Check if the current token can start a new statement in a block.
    /// Used to make semicolons optional in Sigil's advanced syntax.
    fn can_start_stmt(&self) -> bool {
        // Check for keywords that can be used as identifiers (expressions)
        if let Some(token) = self.current_token() {
            if Self::keyword_as_ident(token).is_some() {
                return true;
            }
        }
        matches!(
            self.current_token(),
            Some(
                Token::Let
                    | Token::If
                    | Token::Match
                    | Token::Loop
                    | Token::While
                    | Token::For
                    | Token::Return
                    | Token::Break
                    | Token::Continue
                    | Token::Ident(_)
                    | Token::SelfLower
                    | Token::SelfUpper
                    | Token::LParen
                    | Token::LBracket
                    | Token::LBrace
                    // Literals can start expression statements (e.g., final return value)
                    | Token::StringLit(_)
                    | Token::IntLit(_)
                    | Token::FloatLit(_)
                    | Token::True
                    | Token::False
                    | Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_)
                    | Token::DocComment(_)
            )
        ) || self.can_start_item()
    }

    /// Consume semicolon if present, or skip if next token can start an item.
    /// This makes semicolons optional in Sigil's advanced syntax.
    fn expect_semi_or_item_start(&mut self) -> ParseResult<()> {
        if self.consume_if(&Token::Semi) {
            return Ok(());
        }
        if self.can_start_item() || self.is_eof() || self.check(&Token::RBrace) {
            // Semicolon is optional before a new item, EOF, or closing brace
            return Ok(());
        }
        let span = self.current_span();
        Err(ParseError::UnexpectedToken {
            expected: "`;` or new item".to_string(),
            found: self.current_token().cloned().unwrap_or(Token::Semi),
            span,
        })
    }

    // === Item parsing ===

    fn parse_item(&mut self) -> ParseResult<Spanned<Item>> {
        let start_span = self.current_span();

        // Collect outer attributes (#[...] or @[...])
        let mut outer_attrs = Vec::new();
        while self.check(&Token::Hash) || self.check(&Token::At) {
            outer_attrs.push(self.parse_outer_attribute()?);
        }

        let visibility = self.parse_visibility()?;

        let item = match self.current_token() {
            Some(Token::Fn) | Some(Token::Async) => {
                Item::Function(self.parse_function_with_attrs(visibility, outer_attrs)?)
            }
            Some(Token::Struct) => {
                Item::Struct(self.parse_struct_with_attrs(visibility, outer_attrs)?)
            }
            Some(Token::Enum) => Item::Enum(self.parse_enum(visibility)?),
            Some(Token::Trait) => Item::Trait(self.parse_trait(visibility)?),
            Some(Token::Impl) => Item::Impl(self.parse_impl()?),
            Some(Token::Unsafe) => {
                // unsafe impl, unsafe fn, unsafe trait
                self.advance(); // consume 'unsafe'
                match self.current_token() {
                    Some(Token::Impl) => Item::Impl(self.parse_impl()?),
                    Some(Token::Fn) | Some(Token::Async) => {
                        Item::Function(self.parse_function_with_attrs(visibility, outer_attrs)?)
                    }
                    Some(Token::Trait) => Item::Trait(self.parse_trait(visibility)?),
                    Some(t) => {
                        return Err(ParseError::UnexpectedToken {
                            expected: "impl, fn, or trait after unsafe".to_string(),
                            found: t.clone(),
                            span: self.current_span(),
                        })
                    }
                    None => return Err(ParseError::UnexpectedEof),
                }
            }
            Some(Token::Type) => Item::TypeAlias(self.parse_type_alias(visibility)?),
            Some(Token::Mod) => Item::Module(self.parse_module(visibility)?),
            Some(Token::Use) => Item::Use(self.parse_use(visibility)?),
            Some(Token::Const) => {
                // Check if this is `const fn` (const function) or just `const X: T = ...`
                if self
                    .peek_next()
                    .map(|t| matches!(t, Token::Fn | Token::Async))
                    == Some(true)
                {
                    Item::Function(self.parse_function_with_attrs(visibility, outer_attrs)?)
                } else {
                    Item::Const(self.parse_const(visibility)?)
                }
            }
            Some(Token::Static) => Item::Static(self.parse_static(visibility)?),
            Some(Token::Actor) => Item::Actor(self.parse_actor(visibility)?),
            Some(Token::Extern) => Item::ExternBlock(self.parse_extern_block()?),
            Some(Token::Macro) | Some(Token::MacroRules) => {
                Item::Macro(self.parse_macro_def(visibility)?)
            }
            Some(Token::Naked) => {
                // naked fn -> function with naked attribute
                Item::Function(self.parse_function_with_attrs(visibility, outer_attrs)?)
            }
            Some(Token::Packed) => {
                // packed struct -> struct with packed attribute
                Item::Struct(self.parse_struct_with_attrs(visibility, outer_attrs)?)
            }
            // Plurality items (DAEMONIORUM extensions)
            Some(Token::Alter) => {
                use crate::plurality::PluralityParser;
                Item::Plurality(crate::plurality::PluralityItem::Alter(
                    self.parse_alter_def(visibility)?,
                ))
            }
            Some(Token::Headspace) => {
                use crate::plurality::PluralityParser;
                Item::Plurality(crate::plurality::PluralityItem::Headspace(
                    self.parse_headspace_def(visibility)?,
                ))
            }
            Some(Token::Reality) => {
                use crate::plurality::PluralityParser;
                Item::Plurality(crate::plurality::PluralityItem::Reality(
                    self.parse_reality_def(visibility)?,
                ))
            }
            Some(Token::CoCon) => {
                use crate::plurality::PluralityParser;
                Item::Plurality(crate::plurality::PluralityItem::CoConChannel(
                    self.parse_cocon_channel()?,
                ))
            }
            Some(Token::On) => {
                // Check if this is a trigger handler: "on trigger ..."
                if self.peek_next() == Some(&Token::Trigger) {
                    use crate::plurality::PluralityParser;
                    Item::Plurality(crate::plurality::PluralityItem::TriggerHandler(
                        self.parse_trigger_handler()?,
                    ))
                } else {
                    return Err(ParseError::UnexpectedToken {
                        expected: "item".to_string(),
                        found: Token::On,
                        span: self.current_span(),
                    });
                }
            }
            // Macro invocation at item level: `name! { ... }` or `path::to::macro! { ... }`
            Some(Token::Ident(_)) => {
                // Check if this is a macro invocation (next token after possible path is !)
                if self.looks_like_macro_invocation() {
                    Item::MacroInvocation(self.parse_macro_invocation()?)
                } else {
                    return Err(ParseError::UnexpectedToken {
                        expected: "item".to_string(),
                        found: self.current_token().unwrap().clone(),
                        span: self.current_span(),
                    });
                }
            }
            Some(token) => {
                return Err(ParseError::UnexpectedToken {
                    expected: "item".to_string(),
                    found: token.clone(),
                    span: self.current_span(),
                });
            }
            None => return Err(ParseError::UnexpectedEof),
        };

        let end_span = self.current_span();
        Ok(Spanned::new(item, start_span.merge(end_span)))
    }

    pub(crate) fn parse_visibility(&mut self) -> ParseResult<Visibility> {
        if self.consume_if(&Token::Pub) {
            Ok(Visibility::Public)
        } else {
            Ok(Visibility::Private)
        }
    }

    fn parse_function(&mut self, visibility: Visibility) -> ParseResult<Function> {
        self.parse_function_with_attrs(visibility, Vec::new())
    }

    fn parse_function_with_attrs(
        &mut self,
        visibility: Visibility,
        outer_attrs: Vec<Attribute>,
    ) -> ParseResult<Function> {
        // Parse function attributes from outer attributes
        let mut attrs = self.process_function_attrs(&outer_attrs);

        // Check for naked keyword before fn
        if self.consume_if(&Token::Naked) {
            attrs.naked = true;
        }

        // Check for unsafe keyword before fn
        let is_unsafe = self.consume_if(&Token::Unsafe);

        // Check for const keyword before fn
        let is_const = self.consume_if(&Token::Const);

        let is_async = self.consume_if(&Token::Async);
        self.expect(Token::Fn)?;

        let mut name = self.parse_ident()?;

        // Parse optional evidentiality marker on function name: fn load~<T>() or fn predict◊()
        // parse_ident only consumes unambiguous markers (~, ◊, ‽), so we also check for ! and ?
        if let Some(ev) = self.parse_evidentiality_opt() {
            // Store in the name's evidentiality field if not already set
            if name.evidentiality.is_none() {
                name.evidentiality = Some(ev);
            }
        }

        // Parse optional aspect suffix: ·ing, ·ed, ·able, ·ive
        let aspect = match self.current_token() {
            Some(Token::AspectProgressive) => {
                self.advance();
                Some(Aspect::Progressive)
            }
            Some(Token::AspectPerfective) => {
                self.advance();
                Some(Aspect::Perfective)
            }
            Some(Token::AspectPotential) => {
                self.advance();
                Some(Aspect::Potential)
            }
            Some(Token::AspectResultative) => {
                self.advance();
                Some(Aspect::Resultative)
            }
            _ => None,
        };

        let generics = self.parse_generics_opt()?;

        self.expect(Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(Token::RParen)?;

        let return_type = if self.consume_if(&Token::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        // Handle async marker after return type: -> Type⌛
        // This is an alternative async syntax: fn foo() -> Result⌛ { ... }
        let is_async = is_async || self.consume_if(&Token::Hourglass);

        let where_clause = self.parse_where_clause_opt()?;

        let body = if self.check(&Token::LBrace) {
            Some(self.parse_block()?)
        } else {
            // Semicolon is optional for trait method signatures when followed by
            // another item, a doc comment, or closing brace (Sigil style)
            if !self.consume_if(&Token::Semi) {
                // If no semicolon, we must be at a valid termination point:
                // - Next function/const/type declaration
                // - Doc comment (next trait item)
                // - Closing brace (end of trait/impl)
                // Otherwise it's an error
                let valid_terminator = matches!(
                    self.current_token(),
                    Some(Token::Fn)
                        | Some(Token::Async)
                        | Some(Token::Unsafe)
                        | Some(Token::Const)
                        | Some(Token::Type)
                        | Some(Token::Pub)
                        | Some(Token::DocComment(_))
                        | Some(Token::LineComment(_))
                        | Some(Token::BlockComment(_))
                        | Some(Token::TildeComment(_))
                        | Some(Token::RBrace)
                        | Some(Token::Hash)
                );
                if !valid_terminator {
                    return match self.current_token().cloned() {
                        Some(token) => Err(ParseError::UnexpectedToken {
                            expected: "Semi".to_string(),
                            found: token,
                            span: self.current_span(),
                        }),
                        None => Err(ParseError::UnexpectedEof),
                    };
                }
            }
            None
        };

        Ok(Function {
            visibility,
            is_async,
            is_const,
            is_unsafe,
            attrs,
            name,
            aspect,
            generics,
            params,
            return_type,
            where_clause,
            body,
        })
    }

    /// Process outer attributes into FunctionAttrs.
    fn process_function_attrs(&self, attrs: &[Attribute]) -> FunctionAttrs {
        let mut func_attrs = FunctionAttrs::default();

        for attr in attrs {
            match attr.name.name.as_str() {
                "panic_handler" => func_attrs.panic_handler = true,
                "entry" => func_attrs.entry = true,
                "no_mangle" => func_attrs.no_mangle = true,
                "export" => func_attrs.export = true,
                "cold" => func_attrs.cold = true,
                "hot" => func_attrs.hot = true,
                "test" => func_attrs.test = true,
                "naked" => func_attrs.naked = true,
                "inline" => {
                    func_attrs.inline = Some(match &attr.args {
                        Some(AttrArgs::Paren(args)) => {
                            if let Some(AttrArg::Ident(ident)) = args.first() {
                                match ident.name.as_str() {
                                    "always" => InlineHint::Always,
                                    "never" => InlineHint::Never,
                                    _ => InlineHint::Hint,
                                }
                            } else {
                                InlineHint::Hint
                            }
                        }
                        _ => InlineHint::Hint,
                    });
                }
                "link_section" => {
                    if let Some(AttrArgs::Eq(value)) = &attr.args {
                        if let Expr::Literal(Literal::String(s)) = value.as_ref() {
                            func_attrs.link_section = Some(s.clone());
                        }
                    }
                }
                "interrupt" => {
                    if let Some(AttrArgs::Paren(args)) = &attr.args {
                        if let Some(AttrArg::Literal(Literal::Int { value, base, .. })) =
                            args.first()
                        {
                            let num = Self::parse_int_value(value, *base) as u32;
                            func_attrs.interrupt = Some(num);
                        }
                    }
                }
                "align" => {
                    if let Some(AttrArgs::Paren(args)) = &attr.args {
                        if let Some(AttrArg::Literal(Literal::Int { value, base, .. })) =
                            args.first()
                        {
                            let align = Self::parse_int_value(value, *base) as usize;
                            func_attrs.align = Some(align);
                        }
                    }
                }
                _ => {
                    // Store unrecognized attributes
                    func_attrs.outer_attrs.push(attr.clone());
                }
            }
        }

        func_attrs
    }

    fn parse_struct_with_attrs(
        &mut self,
        visibility: Visibility,
        outer_attrs: Vec<Attribute>,
    ) -> ParseResult<StructDef> {
        // Parse struct attributes
        let mut attrs = StructAttrs::default();
        attrs.outer_attrs = outer_attrs.clone();

        // Process derive attributes
        for attr in &outer_attrs {
            if attr.name.name == "derive" {
                if let Some(AttrArgs::Paren(args)) = &attr.args {
                    for arg in args {
                        if let AttrArg::Ident(ident) = arg {
                            let derive = Self::parse_derive_trait(&ident.name)?;
                            attrs.derives.push(derive);
                        }
                    }
                }
            } else if attr.name.name == "simd" {
                attrs.simd = true;
            } else if attr.name.name == "repr" {
                if let Some(AttrArgs::Paren(args)) = &attr.args {
                    for arg in args {
                        if let AttrArg::Ident(ident) = arg {
                            attrs.repr = Some(match ident.name.as_str() {
                                "C" => StructRepr::C,
                                "transparent" => StructRepr::Transparent,
                                "packed" => {
                                    attrs.packed = true;
                                    StructRepr::C // packed implies C repr
                                }
                                other => StructRepr::Int(other.to_string()),
                            });
                        } else if let AttrArg::Nested(nested) = arg {
                            if nested.name.name == "align" {
                                if let Some(AttrArgs::Paren(align_args)) = &nested.args {
                                    if let Some(AttrArg::Literal(Literal::Int { value, .. })) =
                                        align_args.first()
                                    {
                                        if let Ok(n) = value.parse::<usize>() {
                                            attrs.align = Some(n);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check for packed keyword before struct
        if self.consume_if(&Token::Packed) {
            attrs.packed = true;
        }

        self.expect(Token::Struct)?;
        let name = self.parse_ident()?;

        // Evidentiality markers can appear BEFORE or AFTER generics:
        // - struct Foo! { ... } - newtype/verified struct
        // - struct Bar<T>~ { ... } - reported struct
        // - struct Gradient!<S, D, Dev> { ... } - verified struct with generics after evidentiality
        // Check for evidentiality BEFORE generics: struct Type!<...>
        let _evidentiality_before = self.parse_evidentiality_opt();

        let generics = self.parse_generics_opt()?;

        // Also check for evidentiality AFTER generics: struct Type<T>!
        let _evidentiality_after = self.parse_evidentiality_opt();

        // Optional where clause for struct bounds (parsed but not yet stored in AST)
        let _ = self.parse_where_clause_opt()?;

        let fields = if self.check(&Token::LBrace) {
            self.expect(Token::LBrace)?;
            let fields = self.parse_field_defs()?;
            self.expect(Token::RBrace)?;
            StructFields::Named(fields)
        } else if self.check(&Token::LParen) {
            self.expect(Token::LParen)?;
            let types = self.parse_tuple_struct_fields()?;
            self.expect(Token::RParen)?;
            // Semicolon is optional in Sigil's advanced syntax
            self.expect_semi_or_item_start()?;
            StructFields::Tuple(types)
        } else {
            // Semicolon is optional in Sigil's advanced syntax
            self.expect_semi_or_item_start()?;
            StructFields::Unit
        };

        Ok(StructDef {
            visibility,
            attrs,
            name,
            generics,
            fields,
        })
    }

    fn parse_derive_trait(name: &str) -> ParseResult<DeriveTrait> {
        match name {
            "Debug" => Ok(DeriveTrait::Debug),
            "Clone" => Ok(DeriveTrait::Clone),
            "Copy" => Ok(DeriveTrait::Copy),
            "Default" => Ok(DeriveTrait::Default),
            "PartialEq" => Ok(DeriveTrait::PartialEq),
            "Eq" => Ok(DeriveTrait::Eq),
            "PartialOrd" => Ok(DeriveTrait::PartialOrd),
            "Ord" => Ok(DeriveTrait::Ord),
            "Hash" => Ok(DeriveTrait::Hash),
            // ECS traits
            "Component" => Ok(DeriveTrait::Component),
            "Resource" => Ok(DeriveTrait::Resource),
            "Bundle" => Ok(DeriveTrait::Bundle),
            // Serde traits
            "Serialize" => Ok(DeriveTrait::Serialize),
            "Deserialize" => Ok(DeriveTrait::Deserialize),
            // Custom derive
            _ => Ok(DeriveTrait::Custom(name.to_string())),
        }
    }

    fn parse_enum(&mut self, visibility: Visibility) -> ParseResult<EnumDef> {
        self.expect(Token::Enum)?;
        let name = self.parse_ident()?;
        let generics = self.parse_generics_opt()?;

        self.expect(Token::LBrace)?;
        let mut variants = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_eof() {
            // Skip doc comments, line comments, and attributes before variants
            while matches!(
                self.current_token(),
                Some(Token::DocComment(_))
                    | Some(Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_))
            ) {
                self.advance();
            }
            // Skip any outer attributes (#[...] or @[...])
            while self.check(&Token::Hash) || self.check(&Token::At) {
                self.parse_outer_attribute()?;
            }
            // Skip any additional comments after attributes
            while matches!(
                self.current_token(),
                Some(Token::DocComment(_))
                    | Some(Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_))
            ) {
                self.advance();
            }
            if self.check(&Token::RBrace) {
                break;
            }
            variants.push(self.parse_enum_variant()?);
            if !self.consume_if(&Token::Comma) {
                break;
            }
            // Skip trailing comments after comma
            while matches!(
                self.current_token(),
                Some(Token::DocComment(_))
                    | Some(Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_))
            ) {
                self.advance();
            }
        }
        self.expect(Token::RBrace)?;

        Ok(EnumDef {
            visibility,
            name,
            generics,
            variants,
        })
    }

    fn parse_enum_variant(&mut self) -> ParseResult<EnumVariant> {
        let name = self.parse_ident()?;

        let fields = if self.check(&Token::LBrace) {
            self.expect(Token::LBrace)?;
            let fields = self.parse_field_defs()?;
            self.expect(Token::RBrace)?;
            StructFields::Named(fields)
        } else if self.check(&Token::LParen) {
            self.expect(Token::LParen)?;
            let types = self.parse_attributed_type_list()?;
            self.expect(Token::RParen)?;
            StructFields::Tuple(types)
        } else {
            StructFields::Unit
        };

        let discriminant = if self.consume_if(&Token::Eq) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(EnumVariant {
            name,
            fields,
            discriminant,
        })
    }

    fn parse_trait(&mut self, visibility: Visibility) -> ParseResult<TraitDef> {
        self.expect(Token::Trait)?;
        let name = self.parse_ident()?;
        let generics = self.parse_generics_opt()?;

        let supertraits = if self.consume_if(&Token::Colon) {
            self.parse_type_bounds()?
        } else {
            vec![]
        };

        self.expect(Token::LBrace)?;
        let mut items = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_eof() {
            // Skip doc comments and line comments before trait items
            while matches!(
                self.current_token(),
                Some(Token::DocComment(_))
                    | Some(Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_))
            ) {
                self.advance();
            }
            if self.check(&Token::RBrace) {
                break;
            }
            items.push(self.parse_trait_item()?);
        }
        self.expect(Token::RBrace)?;

        Ok(TraitDef {
            visibility,
            name,
            generics,
            supertraits,
            items,
        })
    }

    fn parse_trait_item(&mut self) -> ParseResult<TraitItem> {
        let visibility = self.parse_visibility()?;

        match self.current_token() {
            Some(Token::Fn) | Some(Token::Async) | Some(Token::Unsafe) => {
                Ok(TraitItem::Function(self.parse_function(visibility)?))
            }
            Some(Token::Type) => {
                self.advance();
                let name = self.parse_ident()?;
                let bounds = if self.consume_if(&Token::Colon) {
                    self.parse_type_bounds()?
                } else {
                    vec![]
                };
                self.expect(Token::Semi)?;
                Ok(TraitItem::Type { name, bounds })
            }
            Some(Token::Const) => {
                // Check if this is `const fn` or just `const NAME: TYPE;`
                if self
                    .peek_next()
                    .map(|t| matches!(t, Token::Fn | Token::Async))
                    == Some(true)
                {
                    Ok(TraitItem::Function(self.parse_function(visibility)?))
                } else {
                    self.advance();
                    let name = self.parse_ident()?;
                    self.expect(Token::Colon)?;
                    let ty = self.parse_type()?;
                    self.expect(Token::Semi)?;
                    Ok(TraitItem::Const { name, ty })
                }
            }
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "trait item".to_string(),
                found: token.clone(),
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_impl(&mut self) -> ParseResult<ImplBlock> {
        self.expect(Token::Impl)?;
        let generics = self.parse_generics_opt()?;

        // Parse either `Trait for Type` or just `Type`
        let first_type = self.parse_type()?;

        let (trait_, self_ty) = if self.consume_if(&Token::For) {
            let self_ty = self.parse_type()?;
            let trait_path = match first_type {
                TypeExpr::Path(p) => p,
                _ => return Err(ParseError::Custom("expected trait path".to_string())),
            };
            (Some(trait_path), self_ty)
        } else {
            (None, first_type)
        };

        // Optional where clause for impl bounds (parsed but not yet stored in AST)
        let _ = self.parse_where_clause_opt()?;

        self.expect(Token::LBrace)?;
        let mut items = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_eof() {
            // Skip doc comments, line comments, and attributes before impl items
            while matches!(
                self.current_token(),
                Some(Token::DocComment(_))
                    | Some(Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_))
                    | Some(Token::Hash)
            ) {
                if self.check(&Token::Hash) {
                    // Skip attribute: #[...] or #![...]
                    self.advance();
                    self.consume_if(&Token::Bang);
                    if self.consume_if(&Token::LBracket) {
                        let mut depth = 1;
                        while depth > 0 && !self.is_eof() {
                            match self.current_token() {
                                Some(Token::LBracket) => depth += 1,
                                Some(Token::RBracket) => depth -= 1,
                                _ => {}
                            }
                            self.advance();
                        }
                    }
                } else {
                    self.advance();
                }
            }
            if self.check(&Token::RBrace) {
                break;
            }
            items.push(self.parse_impl_item()?);
        }
        self.expect(Token::RBrace)?;

        Ok(ImplBlock {
            generics,
            trait_,
            self_ty,
            items,
        })
    }

    fn parse_impl_item(&mut self) -> ParseResult<ImplItem> {
        // Parse outer attributes (#[...] or @[...])
        let mut outer_attrs = Vec::new();
        while self.check(&Token::Hash) || self.check(&Token::At) {
            outer_attrs.push(self.parse_outer_attribute()?);
        }

        let visibility = self.parse_visibility()?;

        match self.current_token() {
            Some(Token::Fn) | Some(Token::Async) | Some(Token::Unsafe) => Ok(ImplItem::Function(
                self.parse_function_with_attrs(visibility, outer_attrs)?,
            )),
            Some(Token::Type) => Ok(ImplItem::Type(self.parse_type_alias(visibility)?)),
            Some(Token::Const) => {
                // Check if this is `const fn` or just `const`
                if self
                    .peek_next()
                    .map(|t| matches!(t, Token::Fn | Token::Async))
                    == Some(true)
                {
                    Ok(ImplItem::Function(
                        self.parse_function_with_attrs(visibility, outer_attrs)?,
                    ))
                } else {
                    Ok(ImplItem::Const(self.parse_const(visibility)?))
                }
            }
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "impl item".to_string(),
                found: token.clone(),
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_type_alias(&mut self, visibility: Visibility) -> ParseResult<TypeAlias> {
        self.expect(Token::Type)?;
        let name = self.parse_ident()?;
        let generics = self.parse_generics_opt()?;
        // Handle evidentiality marker after name: type Weights~ = ...
        let _evidentiality = self.parse_evidentiality_opt();
        self.expect(Token::Eq)?;
        let ty = self.parse_type()?;
        // Semicolon is optional for type aliases (especially after inline struct types)
        self.consume_if(&Token::Semi);

        Ok(TypeAlias {
            visibility,
            name,
            generics,
            ty,
        })
    }

    fn parse_module(&mut self, visibility: Visibility) -> ParseResult<Module> {
        self.expect(Token::Mod)?;
        let name = self.parse_ident()?;

        let items = if self.check(&Token::LBrace) {
            self.expect(Token::LBrace)?;
            let mut items = Vec::new();
            while !self.check(&Token::RBrace) && !self.is_eof() {
                // Skip doc comments and line comments before items
                while matches!(
                    self.current_token(),
                    Some(Token::DocComment(_))
                        | Some(
                            Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_)
                        )
                ) {
                    self.advance();
                }
                if self.check(&Token::RBrace) {
                    break;
                }
                items.push(self.parse_item()?);
            }
            self.expect(Token::RBrace)?;
            Some(items)
        } else {
            // Semicolon is optional in Sigil's advanced syntax
            self.expect_semi_or_item_start()?;
            None
        };

        Ok(Module {
            visibility,
            name,
            items,
        })
    }

    fn parse_use(&mut self, visibility: Visibility) -> ParseResult<UseDecl> {
        self.expect(Token::Use)?;
        let tree = self.parse_use_tree()?;
        // Semicolon is optional in Sigil's advanced syntax
        self.expect_semi_or_item_start()?;

        Ok(UseDecl { visibility, tree })
    }

    fn parse_use_tree(&mut self) -> ParseResult<UseTree> {
        if self.consume_if(&Token::Star) {
            return Ok(UseTree::Glob);
        }

        if self.check(&Token::LBrace) {
            self.expect(Token::LBrace)?;
            let mut trees = Vec::new();
            while !self.check(&Token::RBrace) {
                // Skip comments and attributes inside use groups: #[cfg(...)]
                loop {
                    if matches!(
                        self.current_token(),
                        Some(Token::DocComment(_))
                            | Some(
                                Token::LineComment(_)
                                    | Token::TildeComment(_)
                                    | Token::BlockComment(_)
                            )
                    ) {
                        self.advance();
                    } else if self.check(&Token::Hash) {
                        // Skip attribute: #[...]
                        self.skip_attribute()?;
                    } else {
                        break;
                    }
                }
                if self.check(&Token::RBrace) {
                    break;
                }
                trees.push(self.parse_use_tree()?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
                // Skip comments and attributes after comma
                loop {
                    if matches!(
                        self.current_token(),
                        Some(Token::DocComment(_))
                            | Some(
                                Token::LineComment(_)
                                    | Token::TildeComment(_)
                                    | Token::BlockComment(_)
                            )
                    ) {
                        self.advance();
                    } else if self.check(&Token::Hash) {
                        // Skip attribute: #[...]
                        self.skip_attribute()?;
                    } else {
                        break;
                    }
                }
            }
            self.expect(Token::RBrace)?;
            return Ok(UseTree::Group(trees));
        }

        // Handle special keywords that can appear as path segments: crate, super, self
        let name = if self.check(&Token::Crate) {
            let span = self.current_span();
            self.advance();
            Ident {
                name: "crate".to_string(),
                evidentiality: None,
                affect: None,
                span,
            }
        } else if self.check(&Token::Super) {
            let span = self.current_span();
            self.advance();
            Ident {
                name: "super".to_string(),
                evidentiality: None,
                affect: None,
                span,
            }
        } else if self.check(&Token::SelfLower) {
            let span = self.current_span();
            self.advance();
            Ident {
                name: "self".to_string(),
                evidentiality: None,
                affect: None,
                span,
            }
        } else if self.check(&Token::Sqrt) {
            // Handle sacred constant √2, √3, etc.
            let span = self.current_span();
            self.advance();
            // Check for following integer (e.g., √2)
            if let Some(Token::IntLit(n)) = self.current_token().cloned() {
                let merged_span = span.merge(self.current_span());
                self.advance();
                Ident {
                    name: format!("√{}", n),
                    evidentiality: None,
                    affect: None,
                    span: merged_span,
                }
            } else {
                Ident {
                    name: "√".to_string(),
                    evidentiality: None,
                    affect: None,
                    span,
                }
            }
        } else if self.check(&Token::Phi) {
            // Handle φ (golden ratio) as import name
            let span = self.current_span();
            self.advance();
            Ident {
                name: "φ".to_string(),
                evidentiality: None,
                affect: None,
                span,
            }
        } else {
            self.parse_ident()?
        };

        // Handle macro import or evidentiality suffix: Gradients!, MomentEstimate~
        // These are valid in use statements for macro types or evidential type aliases
        let _evidentiality = self.parse_evidentiality_opt();

        // Check for path continuation with · or ::
        if self.consume_if(&Token::MiddleDot) || self.consume_if(&Token::ColonColon) {
            let suffix = self.parse_use_tree()?;
            return Ok(UseTree::Path {
                prefix: name,
                suffix: Box::new(suffix),
            });
        }

        // Check for rename
        if self.consume_if(&Token::As) {
            // Allow underscore as alias: `use foo::Bar as _`
            if self.check(&Token::Underscore) {
                let span = self.current_span();
                self.advance();
                let alias = Ident {
                    name: "_".to_string(),
                    evidentiality: None,
                    affect: None,
                    span,
                };
                return Ok(UseTree::Rename { name, alias });
            }
            let alias = self.parse_ident()?;
            return Ok(UseTree::Rename { name, alias });
        }

        Ok(UseTree::Name(name))
    }

    fn parse_const(&mut self, visibility: Visibility) -> ParseResult<ConstDef> {
        self.expect(Token::Const)?;
        let name = self.parse_ident()?;
        self.expect(Token::Colon)?;
        let ty = self.parse_type()?;
        self.expect(Token::Eq)?;
        let value = self.parse_expr()?;
        // Semicolon is optional in Sigil's advanced syntax
        self.expect_semi_or_item_start()?;

        Ok(ConstDef {
            visibility,
            name,
            ty,
            value,
        })
    }

    fn parse_static(&mut self, visibility: Visibility) -> ParseResult<StaticDef> {
        self.expect(Token::Static)?;
        let mutable = self.consume_if(&Token::Mut);
        let name = self.parse_ident()?;
        self.expect(Token::Colon)?;
        let ty = self.parse_type()?;
        self.expect(Token::Eq)?;
        let value = self.parse_expr()?;
        // Semicolon is optional in Sigil's advanced syntax
        self.expect_semi_or_item_start()?;

        Ok(StaticDef {
            visibility,
            mutable,
            name,
            ty,
            value,
        })
    }

    /// Parse a macro definition: `macro name { ... }` or `macro name(...) { ... }`
    /// Also supports Rust-style: `macro_rules! name { ... }`
    fn parse_macro_def(&mut self, visibility: Visibility) -> ParseResult<MacroDef> {
        // Handle both `macro` and `macro_rules` keywords
        let is_macro_rules = self.check(&Token::MacroRules);
        if is_macro_rules {
            self.advance(); // consume 'macro_rules'
            self.expect(Token::Bang)?; // consume '!'
        } else {
            self.expect(Token::Macro)?;
        }

        let name = self.parse_ident()?;

        // Collect the entire macro body as a string (we don't interpret macros)
        // Could be: macro name { ... } or macro name(...) { ... } or macro name($arg:ty) { ... }
        let mut body = String::new();
        let mut depth = 0;

        // Check for optional parameter list (parentheses)
        if self.check(&Token::LParen) {
            body.push('(');
            self.advance();
            depth = 1;
            while depth > 0 && !self.is_eof() {
                match self.current_token() {
                    Some(Token::LParen) => {
                        depth += 1;
                        body.push('(');
                    }
                    Some(Token::RParen) => {
                        depth -= 1;
                        if depth > 0 {
                            body.push(')');
                        }
                    }
                    Some(tok) => {
                        body.push_str(&format!("{:?} ", tok));
                    }
                    None => break,
                }
                self.advance();
            }
            body.push(')');
        }

        // Expect the body in braces
        self.expect(Token::LBrace)?;
        body.push('{');
        depth = 1;
        while depth > 0 && !self.is_eof() {
            match self.current_token() {
                Some(Token::LBrace) => {
                    depth += 1;
                    body.push('{');
                }
                Some(Token::RBrace) => {
                    depth -= 1;
                    if depth > 0 {
                        body.push('}');
                    }
                }
                Some(Token::LineComment(s)) => {
                    body.push_str(&format!("//{}", s));
                }
                Some(tok) => {
                    body.push_str(&format!("{:?} ", tok));
                }
                None => break,
            }
            self.advance();
        }
        body.push('}');

        Ok(MacroDef {
            visibility,
            name,
            rules: body,
        })
    }

    /// Check if the current position looks like a macro invocation (ident! or path::ident!)
    fn looks_like_macro_invocation(&mut self) -> bool {
        // Check if current is identifier
        if !matches!(self.current_token(), Some(Token::Ident(_))) {
            return false;
        }
        // Scan ahead for pattern: (ident (:: ident)*) !
        // Position 0 = first token after current (peek_n(0)), etc.
        let mut pos = 0;
        loop {
            match self.peek_n(pos) {
                Some(Token::Bang) => return true,
                Some(Token::ColonColon) => {
                    pos += 1;
                    // Next should be an identifier
                    match self.peek_n(pos) {
                        Some(Token::Ident(_)) => {
                            pos += 1;
                            continue;
                        }
                        _ => return false,
                    }
                }
                _ => return false,
            }
        }
    }

    /// Parse a macro invocation: `name! { ... }` or `path::to::macro! { ... }`
    fn parse_macro_invocation(&mut self) -> ParseResult<MacroInvocation> {
        use crate::ast::{MacroDelimiter, MacroInvocation};

        // Parse the path (macro name, potentially with ::)
        let path = self.parse_type_path()?;

        // Expect !
        self.expect(Token::Bang)?;

        // Determine delimiter and parse body
        let (delimiter, open_tok, close_tok) = match self.current_token() {
            Some(Token::LBrace) => (MacroDelimiter::Brace, Token::LBrace, Token::RBrace),
            Some(Token::LParen) => (MacroDelimiter::Paren, Token::LParen, Token::RParen),
            Some(Token::LBracket) => (MacroDelimiter::Bracket, Token::LBracket, Token::RBracket),
            Some(tok) => {
                return Err(ParseError::UnexpectedToken {
                    expected: "macro delimiter ('{', '(', or '[')".to_string(),
                    found: tok.clone(),
                    span: self.current_span(),
                });
            }
            None => return Err(ParseError::UnexpectedEof),
        };

        self.advance(); // consume opening delimiter

        // Collect body tokens as string
        let mut body = String::new();
        let mut depth = 1;

        while depth > 0 && !self.is_eof() {
            let tok = self.current_token().cloned();
            match &tok {
                Some(t) if *t == open_tok => {
                    depth += 1;
                    body.push_str(&format!("{:?} ", t));
                }
                Some(t) if *t == close_tok => {
                    depth -= 1;
                    if depth > 0 {
                        body.push_str(&format!("{:?} ", t));
                    }
                }
                Some(Token::LineComment(s)) => {
                    body.push_str(&format!("//{}\n", s));
                }
                Some(t) => {
                    body.push_str(&format!("{:?} ", t));
                }
                None => break,
            }
            self.advance();
        }

        // For () and [] delimited macros at item level, consume trailing semicolon
        if delimiter != MacroDelimiter::Brace {
            self.consume_if(&Token::Semi);
        }

        Ok(MacroInvocation {
            path,
            delimiter,
            tokens: body,
        })
    }

    fn parse_actor(&mut self, visibility: Visibility) -> ParseResult<ActorDef> {
        self.expect(Token::Actor)?;
        let name = self.parse_ident()?;
        let generics = self.parse_generics_opt()?;

        self.expect(Token::LBrace)?;

        let mut state = Vec::new();
        let mut handlers = Vec::new();

        while !self.check(&Token::RBrace) && !self.is_eof() {
            if self.check(&Token::On) {
                handlers.push(self.parse_message_handler()?);
            } else {
                // Parse state field
                let vis = self.parse_visibility()?;
                let field_name = self.parse_ident()?;
                self.expect(Token::Colon)?;
                let ty = self.parse_type()?;

                // Optional default value
                let default = if self.consume_if(&Token::Eq) {
                    Some(self.parse_expr()?)
                } else {
                    None
                };

                if !self.check(&Token::RBrace) && !self.check(&Token::On) {
                    self.consume_if(&Token::Comma);
                }

                state.push(FieldDef {
                    visibility: vis,
                    name: field_name,
                    ty,
                    default,
                });
            }
        }

        self.expect(Token::RBrace)?;

        Ok(ActorDef {
            visibility,
            name,
            generics,
            state,
            handlers,
        })
    }

    /// Parse an extern block: `extern "C" { ... }`
    fn parse_extern_block(&mut self) -> ParseResult<ExternBlock> {
        self.expect(Token::Extern)?;

        // Parse ABI string (default to "C")
        let abi = if let Some(Token::StringLit(s)) = self.current_token().cloned() {
            self.advance();
            s
        } else {
            "C".to_string()
        };

        self.expect(Token::LBrace)?;

        let mut items = Vec::new();

        while !self.check(&Token::RBrace) && !self.is_eof() {
            let visibility = self.parse_visibility()?;

            match self.current_token() {
                Some(Token::Fn) => {
                    items.push(ExternItem::Function(
                        self.parse_extern_function(visibility)?,
                    ));
                }
                Some(Token::Static) => {
                    items.push(ExternItem::Static(self.parse_extern_static(visibility)?));
                }
                Some(token) => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "fn or static".to_string(),
                        found: token.clone(),
                        span: self.current_span(),
                    });
                }
                None => return Err(ParseError::UnexpectedEof),
            }
        }

        self.expect(Token::RBrace)?;

        Ok(ExternBlock { abi, items })
    }

    /// Parse an extern function declaration (no body).
    fn parse_extern_function(&mut self, visibility: Visibility) -> ParseResult<ExternFunction> {
        self.expect(Token::Fn)?;
        let name = self.parse_ident()?;

        // Handle evidentiality marker after function name: fn load_safetensors~(...)
        let _evidentiality = self.parse_evidentiality_opt();

        self.expect(Token::LParen)?;

        let mut params = Vec::new();
        let mut variadic = false;

        while !self.check(&Token::RParen) && !self.is_eof() {
            // Check for variadic: ...
            if self.check(&Token::DotDot) {
                self.advance();
                if self.consume_if(&Token::Dot) {
                    variadic = true;
                    break;
                }
            }

            let pattern = self.parse_pattern()?;
            self.expect(Token::Colon)?;
            let ty = self.parse_type()?;

            params.push(Param { pattern, ty });

            if !self.check(&Token::RParen) {
                self.expect(Token::Comma)?;
            }
        }

        self.expect(Token::RParen)?;

        // Return type
        let return_type = if self.consume_if(&Token::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        // Extern functions end with semicolon, not a body
        self.expect(Token::Semi)?;

        Ok(ExternFunction {
            visibility,
            name,
            params,
            return_type,
            variadic,
        })
    }

    /// Parse an extern static declaration.
    fn parse_extern_static(&mut self, visibility: Visibility) -> ParseResult<ExternStatic> {
        self.expect(Token::Static)?;
        let mutable = self.consume_if(&Token::Mut);
        let name = self.parse_ident()?;
        self.expect(Token::Colon)?;
        let ty = self.parse_type()?;
        self.expect(Token::Semi)?;

        Ok(ExternStatic {
            visibility,
            mutable,
            name,
            ty,
        })
    }

    fn parse_message_handler(&mut self) -> ParseResult<MessageHandler> {
        self.expect(Token::On)?;
        let message = self.parse_ident()?;

        self.expect(Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(Token::RParen)?;

        let return_type = if self.consume_if(&Token::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;

        Ok(MessageHandler {
            message,
            params,
            return_type,
            body,
        })
    }

    // === Type parsing ===

    pub(crate) fn parse_type(&mut self) -> ParseResult<TypeExpr> {
        // Check for PREFIX evidentiality: !T, ?T, ~T, ‽T (Sigil-style)
        if let Some(ev) = self.parse_evidentiality_prefix_opt() {
            let inner = self.parse_type()?;
            return Ok(TypeExpr::Evidential {
                inner: Box::new(inner),
                evidentiality: ev,
                error_type: None,
            });
        }

        let mut base = self.parse_type_base()?;

        // Check if parse_ident already consumed evidentiality on the type name (e.g., Tensor◊)
        // If so, we need to extract it and wrap the type properly
        // But DON'T return early - there may be additional evidentiality after generics (e.g., Type~<T>!)
        let path_evidentiality = if let TypeExpr::Path(ref mut path) = base {
            if let Some(last_seg) = path.segments.last_mut() {
                last_seg.ident.evidentiality.take()
            } else {
                None
            }
        } else {
            None
        };
        if let Some(ev) = path_evidentiality {
            base = TypeExpr::Evidential {
                inner: Box::new(base),
                evidentiality: ev,
                error_type: None,
            };
        }

        // Check for evidentiality suffix: T?, T!, T~, T?[Error], T![Error], T~[Error]
        if let Some(ev) = self.parse_evidentiality_opt() {
            // Check for generics after evidentiality: Type!<T>, Type?<T>
            // Pattern: Gradient!<S, D, Dev> where ! is evidentiality, <...> are generics
            let base = if self.check(&Token::Lt) && self.peek_looks_like_generic_arg() {
                // Apply generics to the inner path type
                if let TypeExpr::Path(mut path) = base {
                    self.advance(); // consume <
                    let types = self.parse_type_list()?;
                    self.expect_gt()?;
                    // Add generics to the last segment
                    if let Some(last) = path.segments.last_mut() {
                        last.generics = Some(types);
                    }
                    TypeExpr::Path(path)
                } else {
                    base
                }
            } else {
                base
            };

            // Check for optional error type bracket: ?[ErrorType]
            let error_type = if self.check(&Token::LBracket) {
                self.advance(); // consume [
                let err_ty = self.parse_type()?;
                self.expect(Token::RBracket)?; // consume ]
                Some(Box::new(err_ty))
            } else {
                None
            };

            let mut result = TypeExpr::Evidential {
                inner: Box::new(base),
                evidentiality: ev,
                error_type,
            };

            // Check for additional evidentiality suffix: Type~<T>! (chained evidentiality)
            if let Some(ev2) = self.parse_evidentiality_opt() {
                result = TypeExpr::Evidential {
                    inner: Box::new(result),
                    evidentiality: ev2,
                    error_type: None,
                };
            }

            return Ok(result);
        }

        Ok(base)
    }

    /// Parse PREFIX evidentiality marker: !T, ?T, ~T, ◊T, ‽T
    /// Only consumes the token if followed by valid type start
    fn parse_evidentiality_prefix_opt(&mut self) -> Option<Evidentiality> {
        match self.current_token() {
            Some(Token::Bang) => {
                // Check if this is prefix evidentiality or something else
                // Peek to see if next token could be a type
                if self.peek_is_type_start() {
                    self.advance();
                    Some(Evidentiality::Known)
                } else {
                    None
                }
            }
            Some(Token::Question) => {
                if self.peek_is_type_start() {
                    self.advance();
                    Some(Evidentiality::Uncertain)
                } else {
                    None
                }
            }
            Some(Token::Tilde) => {
                if self.peek_is_type_start() {
                    self.advance();
                    Some(Evidentiality::Reported)
                } else {
                    None
                }
            }
            Some(Token::Lozenge) => {
                if self.peek_is_type_start() {
                    self.advance();
                    Some(Evidentiality::Predicted)
                } else {
                    None
                }
            }
            Some(Token::Interrobang) => {
                if self.peek_is_type_start() {
                    self.advance();
                    Some(Evidentiality::Paradox)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if the next token (after current) could start a type
    fn peek_is_type_start(&mut self) -> bool {
        match self.peek_next() {
            Some(Token::Ident(_)) => true,
            Some(Token::SelfUpper) => true,
            Some(Token::Amp) => true,
            Some(Token::AndAnd) => true, // Double reference: &&T
            Some(Token::Star) => true,
            Some(Token::LBracket) => true,
            Some(Token::LParen) => true,
            Some(Token::Fn) => true,
            Some(Token::Underscore) => true,
            Some(Token::Simd) => true,
            Some(Token::Atomic) => true,
            // Nested prefix evidentiality: !!T, !?T, etc.
            Some(Token::Bang) => true,
            Some(Token::Question) => true,
            Some(Token::Tilde) => true,
            Some(Token::Interrobang) => true,
            _ => false,
        }
    }

    fn parse_type_base(&mut self) -> ParseResult<TypeExpr> {
        match self.current_token() {
            Some(Token::AndAnd) => {
                // Double reference: &&T -> &(&T)
                self.advance();
                // Check for optional lifetime on inner reference: &&'a T
                let lifetime = if let Some(Token::Lifetime(lt)) = self.current_token().cloned() {
                    self.advance();
                    Some(lt)
                } else {
                    None
                };
                let mutable = self.consume_if(&Token::Mut);
                let inner = self.parse_type()?;
                // Inner reference
                let inner_ref = TypeExpr::Reference {
                    lifetime,
                    mutable,
                    inner: Box::new(inner),
                };
                // Outer reference (immutable, no lifetime)
                Ok(TypeExpr::Reference {
                    lifetime: None,
                    mutable: false,
                    inner: Box::new(inner_ref),
                })
            }
            Some(Token::Amp) => {
                self.advance();
                // Check for optional lifetime: &'a T or &'static T
                let lifetime = if let Some(Token::Lifetime(lt)) = self.current_token().cloned() {
                    self.advance();
                    Some(lt)
                } else {
                    None
                };
                let mutable = self.consume_if(&Token::Mut);
                let inner = self.parse_type()?;
                Ok(TypeExpr::Reference {
                    lifetime,
                    mutable,
                    inner: Box::new(inner),
                })
            }
            Some(Token::Star) => {
                self.advance();
                // Raw pointer: *const T, *mut T, or *!T (evidential pointer)
                // Make const/mut optional to support Sigil's *!T syntax
                let mutable = if self.consume_if(&Token::Const) {
                    false
                } else if self.consume_if(&Token::Mut) {
                    true
                } else {
                    // No const/mut - default to immutable (for *!T style)
                    false
                };
                let inner = self.parse_type()?;
                Ok(TypeExpr::Pointer {
                    mutable,
                    inner: Box::new(inner),
                })
            }
            Some(Token::Linear) => {
                // Linear type: `linear T` - value can only be used once
                self.advance();
                let inner = self.parse_type()?;
                Ok(TypeExpr::Linear(Box::new(inner)))
            }
            Some(Token::LBracket) => {
                self.advance();
                // Check for empty brackets or array/shape syntax
                if self.check(&Token::RBracket) {
                    // Empty array/scalar shape: [] - used for scalar tensors (0-dimensional)
                    self.advance();
                    // Return as an empty const expression (empty array literal)
                    return Ok(TypeExpr::ConstExpr(Box::new(Expr::Array(vec![]))));
                }
                // Use parse_type_or_lifetime to handle const generic literals like [2, 4]
                let first = self.parse_type_or_lifetime()?;
                if self.consume_if(&Token::Semi) {
                    // Fixed-size array: [Type; size]
                    let size = self.parse_expr()?;
                    self.expect(Token::RBracket)?;
                    Ok(TypeExpr::Array {
                        element: Box::new(first),
                        size: Box::new(size),
                    })
                } else if self.consume_if(&Token::Comma) {
                    // Multi-element array shape: [A, B, C] or [D, HD/2] for tensor shapes
                    // Parse as const expressions to support arithmetic like HD/2
                    let first_expr = match first {
                        TypeExpr::Path(path) => Expr::Path(path),
                        TypeExpr::ConstExpr(expr) => *expr,
                        _ => Expr::Path(TypePath {
                            segments: vec![PathSegment {
                                ident: Ident {
                                    name: format!("{:?}", first),
                                    evidentiality: None,
                                    affect: None,
                                    span: Span::new(0, 0),
                                },
                                generics: None,
                            }],
                        }),
                    };
                    let mut elem_exprs = vec![first_expr];
                    while !self.check(&Token::RBracket) && !self.is_eof() {
                        // Parse dimension as const expression to support HD/2, N*2, etc.
                        let dim_expr = self.parse_array_dim_expr()?;
                        elem_exprs.push(dim_expr);
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RBracket)?;
                    Ok(TypeExpr::ConstExpr(Box::new(Expr::Array(elem_exprs))))
                } else if matches!(
                    self.current_token(),
                    Some(Token::Slash)
                        | Some(Token::Star)
                        | Some(Token::Plus)
                        | Some(Token::Minus)
                        | Some(Token::Percent)
                ) {
                    // Dimension expression with arithmetic: [HD/2], [H * HD, D], etc.
                    // Convert first to expression and continue parsing as arithmetic
                    let first_expr = match first {
                        TypeExpr::Path(path) => Expr::Path(path),
                        TypeExpr::ConstExpr(expr) => *expr,
                        _ => Expr::Path(TypePath {
                            segments: vec![PathSegment {
                                ident: Ident {
                                    name: format!("{:?}", first),
                                    evidentiality: None,
                                    affect: None,
                                    span: Span::new(0, 0),
                                },
                                generics: None,
                            }],
                        }),
                    };
                    // Now parse the operator and rest of expression
                    let op = match self.current_token() {
                        Some(Token::Slash) => BinOp::Div,
                        Some(Token::Star) => BinOp::Mul,
                        Some(Token::Plus) => BinOp::Add,
                        Some(Token::Minus) => BinOp::Sub,
                        Some(Token::Percent) => BinOp::Rem,
                        _ => unreachable!(),
                    };
                    self.advance(); // consume operator
                    let right = self.parse_const_expr_primary()?;
                    let expr = Expr::Binary {
                        left: Box::new(first_expr),
                        op,
                        right: Box::new(right),
                    };

                    // Check for more dimensions after the arithmetic: [H * HD, D]
                    if self.consume_if(&Token::Comma) {
                        let mut elem_exprs = vec![expr];
                        while !self.check(&Token::RBracket) && !self.is_eof() {
                            let dim_expr = self.parse_array_dim_expr()?;
                            elem_exprs.push(dim_expr);
                            if !self.consume_if(&Token::Comma) {
                                break;
                            }
                        }
                        self.expect(Token::RBracket)?;
                        Ok(TypeExpr::ConstExpr(Box::new(Expr::Array(elem_exprs))))
                    } else {
                        self.expect(Token::RBracket)?;
                        Ok(TypeExpr::ConstExpr(Box::new(Expr::Array(vec![expr]))))
                    }
                } else {
                    // Slice: [Type]
                    self.expect(Token::RBracket)?;
                    Ok(TypeExpr::Slice(Box::new(first)))
                }
            }
            Some(Token::LParen) => {
                self.advance();
                if self.check(&Token::RParen) {
                    self.advance();
                    return Ok(TypeExpr::Tuple(vec![]));
                }
                let types = self.parse_type_list()?;
                self.expect(Token::RParen)?;
                Ok(TypeExpr::Tuple(types))
            }
            Some(Token::Fn) => {
                self.advance();
                self.expect(Token::LParen)?;
                let params = self.parse_type_list()?;
                self.expect(Token::RParen)?;
                let return_type = if self.consume_if(&Token::Arrow) {
                    Some(Box::new(self.parse_type()?))
                } else {
                    None
                };
                Ok(TypeExpr::Function {
                    params,
                    return_type,
                })
            }
            Some(Token::Impl) => {
                // impl Trait - opaque return type
                self.advance();
                // Parse trait bounds: impl Trait + OtherTrait + 'lifetime
                let bounds = self.parse_type_bounds()?;
                Ok(TypeExpr::ImplTrait(bounds))
            }
            Some(Token::Bang) => {
                self.advance();
                Ok(TypeExpr::Never)
            }
            Some(Token::Underscore) => {
                self.advance();
                Ok(TypeExpr::Infer)
            }
            Some(Token::Lt) => {
                // Qualified path: <Type as Trait>::AssociatedType
                // or: <Type>::AssociatedType (inherent associated type)
                self.advance(); // consume <
                let base_type = self.parse_type()?;

                // Check for "as Trait" clause
                let trait_path = if self.consume_if(&Token::As) {
                    Some(self.parse_type_path()?)
                } else {
                    None
                };

                self.expect_gt()?; // consume >
                self.expect(Token::ColonColon)?; // must have :: after >

                // Parse the associated type/const path
                let mut segments = vec![self.parse_path_segment()?];
                while self.consume_if(&Token::ColonColon) {
                    segments.push(self.parse_path_segment()?);
                }

                Ok(TypeExpr::QualifiedPath {
                    self_type: Box::new(base_type),
                    trait_path,
                    item_path: TypePath { segments },
                })
            }
            Some(Token::SelfUpper) => {
                let span = self.current_span();
                self.advance();
                let mut segments = vec![PathSegment {
                    ident: Ident {
                        name: "Self".to_string(),
                        evidentiality: None,
                        affect: None,
                        span,
                    },
                    generics: None,
                }];
                // Continue parsing path: Self::AssociatedType, Self::Nested::Type
                while self.consume_if(&Token::ColonColon) || self.consume_if(&Token::MiddleDot) {
                    // Check for turbofish: Self::<T>
                    if self.check(&Token::Lt) {
                        self.advance();
                        let types = self.parse_type_list()?;
                        self.expect_gt()?;
                        if let Some(last) = segments.last_mut() {
                            last.generics = Some(types);
                        }
                        continue;
                    }
                    segments.push(self.parse_path_segment()?);
                }
                Ok(TypeExpr::Path(TypePath { segments }))
            }
            Some(Token::Simd) => {
                self.advance();
                self.expect(Token::Lt)?;
                let element = self.parse_type()?;
                self.expect(Token::Comma)?;
                let lanes = match self.current_token() {
                    Some(Token::IntLit(s)) => {
                        let n = s
                            .parse::<u8>()
                            .map_err(|_| ParseError::Custom("invalid lane count".to_string()))?;
                        self.advance();
                        n
                    }
                    _ => return Err(ParseError::Custom("expected lane count".to_string())),
                };
                self.expect_gt()?;
                Ok(TypeExpr::Simd {
                    element: Box::new(element),
                    lanes,
                })
            }
            Some(Token::Atomic) => {
                self.advance();
                self.expect(Token::Lt)?;
                let inner = self.parse_type()?;
                self.expect_gt()?;
                Ok(TypeExpr::Atomic(Box::new(inner)))
            }
            Some(Token::Dyn) => {
                // Parse trait object: dyn Trait or dyn Trait + Send + 'static
                self.advance();
                let bounds = self.parse_type_bounds()?;
                Ok(TypeExpr::TraitObject(bounds))
            }
            Some(Token::Struct) => {
                // Inline struct type: struct { field: Type, ... }
                self.advance();
                self.expect(Token::LBrace)?;
                let mut fields = Vec::new();
                while !self.check(&Token::RBrace) && !self.is_eof() {
                    // Skip comments, doc comments, and attributes
                    while matches!(
                        self.current_token(),
                        Some(Token::DocComment(_))
                            | Some(
                                Token::LineComment(_)
                                    | Token::TildeComment(_)
                                    | Token::BlockComment(_)
                            )
                            | Some(Token::Hash)
                    ) {
                        if self.check(&Token::Hash) {
                            // Skip attribute: #[...]
                            self.advance();
                            if self.consume_if(&Token::LBracket) {
                                let mut depth = 1;
                                while depth > 0 && !self.is_eof() {
                                    match self.current_token() {
                                        Some(Token::LBracket) => depth += 1,
                                        Some(Token::RBracket) => depth -= 1,
                                        _ => {}
                                    }
                                    self.advance();
                                }
                            }
                        } else {
                            self.advance();
                        }
                    }
                    if self.check(&Token::RBrace) {
                        break;
                    }
                    // Parse optional visibility
                    let visibility = self.parse_visibility()?;
                    let name = self.parse_ident()?;
                    self.expect(Token::Colon)?;
                    let ty = self.parse_type()?;
                    fields.push(FieldDef {
                        visibility,
                        name,
                        ty,
                        default: None,
                    });
                    if !self.consume_if(&Token::Comma) {
                        break;
                    }
                }
                self.expect(Token::RBrace)?;
                Ok(TypeExpr::InlineStruct { fields })
            }
            Some(Token::Enum) => {
                // Inline enum type: enum { Variant1, Variant2(Type), ... }
                self.advance();
                self.expect(Token::LBrace)?;
                let mut variants = Vec::new();
                while !self.check(&Token::RBrace) && !self.is_eof() {
                    // Skip comments and doc comments
                    while matches!(
                        self.current_token(),
                        Some(Token::DocComment(_))
                            | Some(
                                Token::LineComment(_)
                                    | Token::TildeComment(_)
                                    | Token::BlockComment(_)
                            )
                    ) {
                        self.advance();
                    }
                    if self.check(&Token::RBrace) {
                        break;
                    }
                    // Parse variant name
                    let name = self.parse_ident()?;
                    // Parse optional fields
                    let fields = if self.check(&Token::LParen) {
                        self.advance();
                        let mut types = Vec::new();
                        while !self.check(&Token::RParen) && !self.is_eof() {
                            types.push(self.parse_type()?);
                            if !self.consume_if(&Token::Comma) {
                                break;
                            }
                        }
                        self.expect(Token::RParen)?;
                        StructFields::Tuple(types)
                    } else if self.check(&Token::LBrace) {
                        self.advance();
                        let mut fields = Vec::new();
                        while !self.check(&Token::RBrace) && !self.is_eof() {
                            let name = self.parse_ident()?;
                            self.expect(Token::Colon)?;
                            let ty = self.parse_type()?;
                            fields.push(FieldDef {
                                visibility: Visibility::Private,
                                name,
                                ty,
                                default: None,
                            });
                            if !self.consume_if(&Token::Comma) {
                                break;
                            }
                        }
                        self.expect(Token::RBrace)?;
                        StructFields::Named(fields)
                    } else {
                        StructFields::Unit
                    };
                    // Parse optional discriminant: N = -1
                    let discriminant = if self.consume_if(&Token::Eq) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    variants.push(EnumVariant {
                        name,
                        fields,
                        discriminant,
                    });
                    if !self.consume_if(&Token::Comma) {
                        break;
                    }
                }
                self.expect(Token::RBrace)?;
                Ok(TypeExpr::InlineEnum { variants })
            }
            // Handle crate::, self::, super:: path prefixes in types
            Some(Token::Crate) | Some(Token::SelfLower) | Some(Token::Super) => {
                let keyword = self.current_token().cloned();
                let span = self.current_span();
                self.advance();

                // Build the first segment from the keyword
                let keyword_name = match keyword {
                    Some(Token::Crate) => "crate",
                    Some(Token::SelfLower) => "self",
                    Some(Token::Super) => "super",
                    _ => unreachable!(),
                };
                let first_segment = PathSegment {
                    ident: Ident {
                        name: keyword_name.to_string(),
                        evidentiality: None,
                        affect: None,
                        span,
                    },
                    generics: None,
                };

                let mut segments = vec![first_segment];

                // Continue parsing path: crate::module::Type
                while self.consume_if(&Token::ColonColon) || self.consume_if(&Token::MiddleDot) {
                    // Check for turbofish: crate::<T> (unlikely but possible)
                    if self.check(&Token::Lt) {
                        self.advance();
                        let types = self.parse_type_list()?;
                        self.expect_gt()?;
                        if let Some(last) = segments.last_mut() {
                            last.generics = Some(types);
                        }
                        continue;
                    }
                    segments.push(self.parse_path_segment()?);
                }
                Ok(TypeExpr::Path(TypePath { segments }))
            }
            _ => {
                let path = self.parse_type_path()?;
                Ok(TypeExpr::Path(path))
            }
        }
    }

    fn parse_type_path(&mut self) -> ParseResult<TypePath> {
        let mut segments = Vec::new();
        segments.push(self.parse_path_segment()?);

        // Don't continue parsing path if there's a pending `>` from split `>>`
        // This handles cases like `<T as Trait<U>>::Assoc` where `>>` is split
        // and the `::Assoc` belongs to the outer qualified path, not the trait path
        while !self.pending_gt.is_some()
            && (self.consume_if(&Token::ColonColon) || self.consume_if(&Token::MiddleDot))
        {
            // Check for turbofish syntax: path::<Type> instead of path::segment
            if self.check(&Token::Lt) {
                // Parse turbofish generics for the last segment
                // Temporarily exit condition context - turbofish is always type context
                let was_in_condition = self.in_condition;
                self.in_condition = false;
                self.advance(); // consume <
                let types = self.parse_type_list()?;
                self.expect_gt()?;
                self.in_condition = was_in_condition;
                // Update the last segment with these generics
                if let Some(last) = segments.last_mut() {
                    last.generics = Some(types);
                }
                // Continue parsing - there may be more segments after turbofish
                // e.g., Option::<T>::None or Vec::<T>::new()
                continue;
            }
            segments.push(self.parse_path_segment()?);
        }

        Ok(TypePath { segments })
    }

    fn parse_path_segment(&mut self) -> ParseResult<PathSegment> {
        // Handle both identifiers and numeric indices (for tuple field access like tuple·0)
        let ident = if let Some(Token::IntLit(idx)) = self.current_token().cloned() {
            let span = self.current_span();
            self.advance();
            Ident {
                name: idx,
                evidentiality: None,
                affect: None,
                span,
            }
        } else {
            self.parse_ident()?
        };

        // Special case: Fn(T) -> R, FnMut(T) -> R, FnOnce(T) -> R trait syntax
        // These are function trait bounds with parenthesis-style generics
        let is_fn_trait = matches!(ident.name.as_str(), "Fn" | "FnMut" | "FnOnce");
        if is_fn_trait && self.check(&Token::LParen) {
            self.advance(); // consume (
            let param_types = self.parse_type_list()?;
            self.expect(Token::RParen)?;
            // Check for optional return type: -> R
            let return_type = if self.consume_if(&Token::Arrow) {
                Some(self.parse_type()?)
            } else {
                None
            };
            // Build a tuple type for params and optional return as generics
            // Fn(A, B) -> R becomes Fn<(A, B), R>
            let mut generics = vec![TypeExpr::Tuple(param_types)];
            if let Some(ret) = return_type {
                generics.push(ret);
            }
            return Ok(PathSegment {
                ident,
                generics: Some(generics),
            });
        }

        // Don't parse generics in condition context (< is comparison, not generics)
        // Also check that what follows < looks like a type, not an expression like `self`
        // Support both <T> and [T] syntax for type generics (Sigil alternative syntax)
        let generics = if !self.is_in_condition()
            && self.check(&Token::Lt)
            && self.peek_looks_like_generic_arg()
        {
            self.advance(); // consume <
            let types = self.parse_type_list()?;
            // Use expect_gt() to handle nested generics with `>>`
            self.expect_gt()?;
            Some(types)
        } else if self.check(&Token::LBracket) && self.peek_looks_like_bracket_generic() {
            // Alternative syntax: Type[T] instead of Type<T>
            self.advance(); // consume [
            let types = self.parse_type_list()?;
            self.expect(Token::RBracket)?;
            Some(types)
        } else {
            None
        };

        Ok(PathSegment { ident, generics })
    }

    /// Check if [...] after identifier looks like a generic type argument rather than an array
    fn peek_looks_like_bracket_generic(&mut self) -> bool {
        // Peek after [ to check if it looks like a type parameter
        // Be conservative - index expressions like `array[pos + 2..]` should not be treated as generics
        match self.peek_next().cloned() {
            // For identifiers, use 2-token lookahead to see if it's a type or expression
            Some(Token::Ident(name)) => {
                // Check what follows the identifier
                // Only treat as generic if the identifier looks like a type name (uppercase)
                let is_type_name = name.chars().next().map_or(false, |c| c.is_uppercase());
                match self.peek_n(1) {
                    // [T] - only treat as generic if T looks like a type name (uppercase)
                    // This distinguishes HashMap[String] (generic) from array[index] (indexing)
                    Some(Token::RBracket) => is_type_name,
                    Some(Token::Comma) => is_type_name, // [T, U] - but not [a, b] which is array
                    Some(Token::ColonColon) => true,    // [T::U]
                    Some(Token::Lt) => is_type_name,    // [T<U>]
                    Some(Token::LBracket) => is_type_name, // [T[U]] but not [a[b]] (nested index)
                    // Evidentiality markers: [T!], [T?], [T~], [T◊], [T‽]
                    Some(Token::Question) => true,
                    Some(Token::Bang) => true,
                    Some(Token::Tilde) => true,
                    Some(Token::Lozenge) => true,
                    Some(Token::Interrobang) => true,
                    // Associated type binding: [Output = Type]
                    Some(Token::Eq) => true,
                    // Expression operators indicate index expression, not generics
                    Some(Token::Plus) => false,     // [pos + 2]
                    Some(Token::Minus) => false,    // [len - 1]
                    Some(Token::Star) => false,     // [i * 2]
                    Some(Token::Slash) => false,    // [i / 2]
                    Some(Token::DotDot) => false,   // [pos..] range
                    Some(Token::DotDotEq) => false, // [0..=n]
                    _ => false,                     // Default to not treating as generics
                }
            }
            Some(Token::SelfUpper) => true, // [Self]
            Some(Token::Amp) => true,       // [&T]
            // Don't treat [*expr] as generic - could be dereference in index
            Some(Token::Star) => false,
            Some(Token::Fn) => true, // [fn(...)]
            Some(Token::LParen) => {
                // [()] could be tuple type generic or parenthesized expression index
                // Look inside the parens to decide:
                // - [()] empty tuple - likely type
                // - [(T, U)] uppercase identifiers - likely type
                // - [(expr)] lowercase identifier or expression - likely index
                match self.peek_n(1) {
                    Some(Token::RParen) => true, // [()] empty tuple type
                    Some(Token::Ident(name)) => {
                        // If identifier starts uppercase, likely a type
                        // If lowercase followed by expression ops like 'as', it's an expression
                        if name.chars().next().map_or(false, |c| c.is_uppercase()) {
                            true // [(Type...)]
                        } else {
                            // Check what follows the lowercase identifier
                            match self.peek_n(2) {
                                Some(Token::As) => false,       // [(n as T)] cast expression
                                Some(Token::Plus) => false,     // [(a + b)]
                                Some(Token::Minus) => false,    // [(a - b)]
                                Some(Token::Star) => false,     // [(a * b)]
                                Some(Token::Slash) => false,    // [(a / b)]
                                Some(Token::Dot) => false,      // [(a.b)]
                                Some(Token::LBracket) => false, // [(a[i])]
                                Some(Token::LParen) => false,   // [(f())]
                                Some(Token::RParen) => false, // [(x)] single lowercase var - expression
                                Some(Token::Comma) => true, // [(a, b)] could be tuple type, try it
                                _ => false,                 // Default to expression (index)
                            }
                        }
                    }
                    _ => false, // Default to not treating as generics
                }
            }
            Some(Token::Dyn) => true,  // [dyn Trait]
            Some(Token::Impl) => true, // [impl Trait]
            // Path-starting keywords that indicate type paths
            Some(Token::Crate) => true, // [crate::Type]
            Some(Token::Super) => true, // [super::Type]
            // Literals indicate expressions, not types
            Some(Token::IntLit(_)) => false,
            Some(Token::FloatLit(_)) => false,
            Some(Token::SelfLower) => false, // [self.x] - expression
            // Could be array: [expr; size] or [type; size]
            // If it's for generics, there won't be a semicolon
            _ => false,
        }
    }

    /// Check if what follows < looks like it could be a generic type argument
    /// This helps disambiguate `foo<T>` (generics) from `foo < bar` (comparison)
    /// We are conservative: only treat as generics if it's clearly a type context
    /// Note: When called, self.current is at the < token
    fn peek_looks_like_generic_arg(&mut self) -> bool {
        // Use peek_next (peek_n(0)) to see what's after <
        match self.peek_next().cloned() {
            // Clear type starts that don't look like expressions
            Some(Token::Amp) => true, // &T - references are type-like
            Some(Token::Star) => {
                // *const T or *mut T - pointer types
                // *expr - dereference (not a type)
                // Look at what follows * to distinguish
                match self.peek_n(1) {
                    Some(Token::Const) => true, // *const T - pointer type
                    Some(Token::Mut) => true,   // *mut T - pointer type
                    _ => false,                 // *expr - dereference, not a type
                }
            }
            Some(Token::LBracket) => true,    // [T] - slices
            Some(Token::LParen) => true,      // () - tuple types including unit
            Some(Token::Fn) => true,          // fn() - function types
            Some(Token::Simd) => true,        // simd<T, N>
            Some(Token::Atomic) => true,      // atomic<T>
            Some(Token::Dyn) => true,         // dyn Trait - trait objects
            Some(Token::Impl) => true,        // impl Trait - existential types
            Some(Token::SelfUpper) => true,   // Self is a type
            Some(Token::Crate) => true,       // crate::Type - path starting with crate
            Some(Token::Super) => true,       // super::Type - path starting with super
            Some(Token::Lifetime(_)) => true, // 'a, 'static - lifetime type args
            Some(Token::Underscore) => true,  // _ - inferred type
            // Evidentiality prefixes on types: !T, ?T, ~T
            Some(Token::Bang) => true,
            Some(Token::Question) => true,
            Some(Token::Tilde) => true,
            Some(Token::Interrobang) => true,
            // Path-rooted types: crate::Type, super::Type
            Some(Token::Crate) => true,
            Some(Token::Super) => true,
            // For identifiers, we need 2-token lookahead to see what follows
            Some(Token::Ident(name)) => {
                // peek_n(1) looks at the token after the identifier
                // (peek_n(0) = token after <, peek_n(1) = token after that)
                let is_type_like = name.chars().next().map_or(false, |c| c.is_uppercase())
                    || matches!(
                        name.as_str(),
                        // Primitive types are lowercase but are type names
                        "u8" | "u16"
                            | "u32"
                            | "u64"
                            | "u128"
                            | "usize"
                            | "i8"
                            | "i16"
                            | "i32"
                            | "i64"
                            | "i128"
                            | "isize"
                            | "f32"
                            | "f64"
                            | "bool"
                            | "char"
                            | "str"
                    );
                match self.peek_n(1) {
                    // Only treat as generic if clearly followed by generic-context tokens
                    Some(Token::Gt) => true,
                    Some(Token::Shr) => true, // >> which may close nested generics
                    // For comma: only treat as generic if identifier looks like a type name
                    // This distinguishes `HashMap<K, V>` (generic) from `x < y,` (comparison in match)
                    Some(Token::Comma) => is_type_like,
                    Some(Token::ColonColon) => true, // T::U path
                    Some(Token::Lt) => true,         // T<U> nested generic
                    Some(Token::LBracket) => true,   // T[U] bracket generic
                    // Evidentiality markers after type name: T?, T!, T~, T◊, T‽
                    Some(Token::Question) => true,
                    Some(Token::Bang) => true,
                    Some(Token::Tilde) => true,
                    Some(Token::Lozenge) => true,
                    Some(Token::Interrobang) => true,
                    // Associated type bindings: <Item = Type>
                    Some(Token::Eq) => true,
                    // Trait bounds: <T: Trait>
                    Some(Token::Colon) => true,
                    _ => false,
                }
            }
            // NOT a type - likely comparison with expression
            Some(Token::SelfLower) => false, // self is an expression
            // Integer literals in generics: const generic values like Type<50257, 1024>
            // Only treat as generic if followed by comma or closing >
            Some(Token::IntLit(_)) => {
                match self.peek_n(1) {
                    Some(Token::Comma) => true, // <50257, 1024, ...>
                    Some(Token::Gt) => true,    // <50257>
                    Some(Token::Shr) => true,   // <50257>> nested
                    _ => false,                 // <5 + x> - comparison
                }
            }
            Some(Token::FloatLit(_)) => false,
            Some(Token::StringLit(_)) => false,
            Some(Token::True) | Some(Token::False) => false,
            Some(Token::Null) => false,
            _ => false, // Default to not parsing as generics if uncertain
        }
    }

    /// Check if the token after | looks like a pipe operation
    /// This distinguishes `expr|τ{...}` (pipe) from `a | b` (bitwise OR)
    fn peek_looks_like_pipe_op(&mut self) -> bool {
        match self.peek_next() {
            // Greek letters for morpheme operations
            Some(Token::Tau) => true,         // |τ{...} transform
            Some(Token::Phi) => true,         // |φ{...} filter
            Some(Token::Sigma) => true,       // |σ sort
            Some(Token::Rho) => true,         // |ρ+ reduce
            Some(Token::Pi) => true,          // |Π product
            Some(Token::Alpha) => true,       // |α first
            Some(Token::Omega) => true,       // |ω last
            Some(Token::Mu) => true,          // |μ middle
            Some(Token::Chi) => true,         // |χ choice
            Some(Token::Nu) => true,          // |ν nth
            Some(Token::Xi) => true,          // |ξ slice
            Some(Token::Delta) => true,       // |δ diff
            Some(Token::Iota) => true,        // |⍳ enumerate
            Some(Token::ForAll) => true,      // |∀ forall
            Some(Token::Exists) => true,      // |∃ exists
            Some(Token::Compose) => true,     // |∘ compose
            Some(Token::Bowtie) => true,      // |⋈ zip/join
            Some(Token::Integral) => true,    // |∫ scan
            Some(Token::Partial) => true,     // |∂ diff
            Some(Token::Nabla) => true,       // |∇ gradient
            Some(Token::GradeUp) => true,     // |⍋ sort ascending
            Some(Token::GradeDown) => true,   // |⍒ sort descending
            Some(Token::Rotate) => true,      // |⌽ reverse
            Some(Token::CycleArrow) => true,  // |↻ cycle
            Some(Token::QuadDiamond) => true, // |⌺ windows
            Some(Token::SquaredPlus) => true, // |⊞ chunks
            Some(Token::ElementSmallVerticalBar) => true, // |⋳ flatten
            Some(Token::Union) => true,       // |∪ unique
            // Keywords for pipe operations
            Some(Token::Match) => true,  // |match{...}
            Some(Token::Send) => true,   // |send{...}
            Some(Token::Recv) => true,   // |recv
            Some(Token::Stream) => true, // |stream{...}
            // Protocol tokens
            Some(Token::ProtoSend) => true,   // |⇒{...}
            Some(Token::ProtoRecv) => true,   // |⇐
            Some(Token::ProtoStream) => true, // |≋{...}
            // Other pipe operation keywords
            Some(Token::Header) => true,      // |header{...}
            Some(Token::Body) => true,        // |body{...}
            Some(Token::Interrobang) => true, // |‽
            // Holographic operators
            Some(Token::Lozenge) => true,     // |◊ or |◊method - possibility
            Some(Token::BoxSymbol) => true,   // |□ or |□method - necessity
            // Identifier could be pipe method: |collect, |take, etc.
            // But identifiers NOT followed by `(` or `{` are likely bitwise OR operands
            Some(Token::Ident(name)) => {
                // Some pipe methods don't require parentheses
                let no_args_pipe_methods = [
                    "collect", "observe", "len", "first", "last", "reverse",
                    "iter", "into_iter", "enumerate", "sum", "product",
                    "min", "max", "count", "flatten", "unique",
                    // Quantum gates
                    "H", "X", "Y", "Z", "S", "T", "measure",
                    "H_all", "measure_all",
                    // Neural network activations and tensor operations
                    "relu", "softmax", "reshape", "backward",
                ];
                if no_args_pipe_methods.contains(&name.as_str()) {
                    return true;
                }
                // Only treat as pipe method if followed by explicit call syntax
                // peek_next() gave us the Ident, peek_n(1) gives us the token after it
                // Also handle evidentiality markers: |validate!{...} where ! precedes {
                let after_ident = self.peek_n(1);
                match after_ident {
                    Some(Token::LParen) | Some(Token::LBrace) => true,
                    // Evidentiality markers followed by call syntax
                    Some(Token::Bang)
                    | Some(Token::Question)
                    | Some(Token::Tilde)
                    | Some(Token::Lozenge) => {
                        matches!(self.peek_n(2), Some(Token::LParen) | Some(Token::LBrace))
                    }
                    _ => false,
                }
            }
            // Reference expression: |&self.field for piping to borrows
            Some(Token::Amp) => true,
            // Direct closure: |{x => body}
            Some(Token::LBrace) => true,
            // Everything else is likely bitwise OR
            _ => false,
        }
    }

    fn parse_type_list(&mut self) -> ParseResult<Vec<TypeExpr>> {
        let mut types = Vec::new();
        // Check for empty list - also check Shr (>>) for nested generics
        if !self.check(&Token::RParen)
            && !self.check(&Token::RBracket)
            && !self.check(&Token::Gt)
            && !self.check(&Token::Shr)
            && self.pending_gt.is_none()
        {
            // Use parse_type_or_lifetime to handle generic args like <'a, T>
            types.push(self.parse_type_or_lifetime()?);
            // Continue parsing more types while we see commas
            // But check for pending_gt BEFORE consuming comma to avoid eating param separators
            while !self.pending_gt.is_some()
                && !self.check(&Token::Gt)
                && !self.check(&Token::Shr)
                && self.consume_if(&Token::Comma)
            {
                // Trailing comma check
                if self.check(&Token::RParen)
                    || self.check(&Token::RBracket)
                    || self.check(&Token::Gt)
                    || self.check(&Token::Shr)
                {
                    break;
                }
                types.push(self.parse_type_or_lifetime()?);
            }
        }
        Ok(types)
    }

    /// Parse a type list that may have attributes before each type
    /// e.g., `(#[from] Error, #[source] io::Error)`
    fn parse_attributed_type_list(&mut self) -> ParseResult<Vec<TypeExpr>> {
        let mut types = Vec::new();
        if !self.check(&Token::RParen) {
            loop {
                // Skip any attributes before the type
                while self.check(&Token::Hash) {
                    self.advance();
                    self.consume_if(&Token::Bang); // for #![...]
                    if self.consume_if(&Token::LBracket) {
                        let mut depth = 1;
                        while depth > 0 && !self.is_eof() {
                            match self.current_token() {
                                Some(Token::LBracket) => depth += 1,
                                Some(Token::RBracket) => depth -= 1,
                                _ => {}
                            }
                            self.advance();
                        }
                    }
                }
                // Parse the type
                types.push(self.parse_type()?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
        }
        Ok(types)
    }

    /// Parse tuple struct fields, which may have optional `pub` visibility before each type
    /// e.g., `struct Foo(pub String, i32)` or `struct Bar(pub(crate) Type)`
    fn parse_tuple_struct_fields(&mut self) -> ParseResult<Vec<TypeExpr>> {
        let mut types = Vec::new();
        if !self.check(&Token::RParen) {
            loop {
                // Skip optional visibility modifier (pub, pub(crate), pub(super), etc.)
                if self.check(&Token::Pub) {
                    self.advance();
                    // Handle pub(crate), pub(super), pub(self), pub(in path)
                    if self.check(&Token::LParen) {
                        self.advance();
                        // Skip tokens until matching RParen
                        let mut depth = 1;
                        while depth > 0 {
                            match self.current_token() {
                                Some(Token::LParen) => depth += 1,
                                Some(Token::RParen) => depth -= 1,
                                None => break,
                                _ => {}
                            }
                            self.advance();
                        }
                    }
                }
                // Parse the type
                types.push(self.parse_type()?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
                // Check for trailing comma
                if self.check(&Token::RParen) {
                    break;
                }
            }
        }
        Ok(types)
    }

    fn parse_type_bounds(&mut self) -> ParseResult<Vec<TypeExpr>> {
        let mut bounds = Vec::new();

        // Handle empty bounds: `T: ,` or `[(); K]: ,` (just checking type is well-formed)
        if self.check(&Token::Comma) || self.check(&Token::LBrace) || self.check(&Token::Semi) {
            return Ok(bounds);
        }

        bounds.push(self.parse_type_or_lifetime()?);
        while self.consume_if(&Token::Plus) {
            bounds.push(self.parse_type_or_lifetime()?);
        }
        Ok(bounds)
    }

    /// Parse either a type or a lifetime (for trait bounds like `T: Trait + 'static`)
    /// Also handles HRTB: `for<'de> Deserialize<'de>`
    /// Also handles associated type bindings: `Output = Type`
    fn parse_type_or_lifetime(&mut self) -> ParseResult<TypeExpr> {
        if let Some(Token::Lifetime(name)) = self.current_token().cloned() {
            self.advance();
            Ok(TypeExpr::Lifetime(name))
        } else if self.check(&Token::For) {
            // Higher-ranked trait bound: for<'a, 'b> Trait<'a, 'b>
            self.advance(); // consume 'for'
            self.expect(Token::Lt)?; // <
            let mut lifetimes = Vec::new();
            if let Some(Token::Lifetime(lt)) = self.current_token().cloned() {
                lifetimes.push(lt);
                self.advance();
                while self.consume_if(&Token::Comma) {
                    if let Some(Token::Lifetime(lt)) = self.current_token().cloned() {
                        lifetimes.push(lt);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect_gt()?; // >
            let bound = self.parse_type()?;
            Ok(TypeExpr::Hrtb {
                lifetimes,
                bound: Box::new(bound),
            })
        } else if matches!(self.current_token(), Some(Token::Ident(_)))
            && self.peek_next() == Some(&Token::Eq)
        {
            // Associated type binding: `Output = Type`
            let name = self.parse_ident()?;
            self.expect(Token::Eq)?;
            let ty = self.parse_type()?;
            Ok(TypeExpr::AssocTypeBinding {
                name,
                ty: Box::new(ty),
            })
        } else if matches!(
            self.current_token(),
            Some(Token::IntLit(_))
                | Some(Token::HexLit(_))
                | Some(Token::BinaryLit(_))
                | Some(Token::OctalLit(_))
        ) {
            // Const generic: numeric literal or expression in type position like `<32>` or `<3 * D>`
            // Parse as expression to handle `3 * D`, `N + 1`, etc.
            let expr = self.parse_const_expr_simple()?;
            Ok(TypeExpr::ConstExpr(Box::new(expr)))
        } else if self.check(&Token::LBrace) {
            // Const block expression: `<{N + 1}>`
            self.advance();
            let expr = self.parse_expr()?;
            self.expect(Token::RBrace)?;
            Ok(TypeExpr::ConstExpr(Box::new(expr)))
        } else {
            self.parse_type()
        }
    }

    /// Parse an array dimension expression like `HD/2`, `N*2`, `MAX_SEQ`, or `{ const { if ... } }`
    /// Used for tensor shape dimensions: [A, B, HD/2]
    fn parse_array_dim_expr(&mut self) -> ParseResult<Expr> {
        // Handle const block: { const { ... } } or just { expr }
        if self.check(&Token::LBrace) {
            self.advance();
            let expr = self.parse_expr()?;
            self.expect(Token::RBrace)?;
            return Ok(expr);
        }
        // Parse as const expression (identifier or literal followed by optional arithmetic)
        self.parse_const_expr_simple()
    }

    /// Parse a simple const expression for use in type positions like `[3 * D, D]`
    /// Handles simple arithmetic: literals, identifiers, and +-*/ operations
    /// Stops at: comma, >, ], ), ;
    fn parse_const_expr_simple(&mut self) -> ParseResult<Expr> {
        let mut lhs = self.parse_const_expr_primary()?;

        loop {
            match self.current_token() {
                Some(Token::Star) => {
                    self.advance();
                    let rhs = self.parse_const_expr_primary()?;
                    lhs = Expr::Binary {
                        op: BinOp::Mul,
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    };
                }
                Some(Token::Plus) => {
                    self.advance();
                    let rhs = self.parse_const_expr_primary()?;
                    lhs = Expr::Binary {
                        op: BinOp::Add,
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    };
                }
                Some(Token::Minus) => {
                    self.advance();
                    let rhs = self.parse_const_expr_primary()?;
                    lhs = Expr::Binary {
                        op: BinOp::Sub,
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    };
                }
                Some(Token::Slash) => {
                    self.advance();
                    let rhs = self.parse_const_expr_primary()?;
                    lhs = Expr::Binary {
                        op: BinOp::Div,
                        left: Box::new(lhs),
                        right: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    /// Parse a primary element for const expressions
    fn parse_const_expr_primary(&mut self) -> ParseResult<Expr> {
        match self.current_token().cloned() {
            Some(Token::IntLit(_))
            | Some(Token::HexLit(_))
            | Some(Token::BinaryLit(_))
            | Some(Token::OctalLit(_)) => {
                let lit = self.parse_literal()?;
                Ok(Expr::Literal(lit))
            }
            Some(Token::Ident(_)) => {
                let path = self.parse_type_path()?;
                Ok(Expr::Path(path))
            }
            Some(Token::Underscore) => {
                // Inferred dimension: [_, N] means first dimension is inferred
                let span = self.current_span();
                self.advance();
                Ok(Expr::Path(TypePath {
                    segments: vec![PathSegment {
                        ident: Ident {
                            name: "_".to_string(),
                            evidentiality: None,
                            affect: None,
                            span,
                        },
                        generics: None,
                    }],
                }))
            }
            Some(Token::LParen) => {
                self.advance();
                let expr = self.parse_const_expr_simple()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            _ => Err(ParseError::Custom("expected const expression".to_string())),
        }
    }

    // === Expression parsing (Pratt parser) ===

    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        // Skip leading comments (line comments, doc comments)
        self.skip_comments();
        let lhs = self.parse_expr_bp(0)?;

        // Check for assignment: expr = value
        if self.consume_if(&Token::Eq) {
            let value = self.parse_expr()?;
            return Ok(Expr::Assign {
                target: Box::new(lhs),
                value: Box::new(value),
            });
        }

        // Check for compound assignment: expr += value, expr -= value, etc.
        // Desugar to: expr = expr op value
        let compound_op = match self.current_token() {
            Some(Token::PlusEq) => Some(BinOp::Add),
            Some(Token::MinusEq) => Some(BinOp::Sub),
            Some(Token::StarEq) => Some(BinOp::Mul),
            Some(Token::SlashEq) => Some(BinOp::Div),
            Some(Token::PercentEq) => Some(BinOp::Rem),
            Some(Token::ShlEq) => Some(BinOp::Shl),
            Some(Token::ShrEq) => Some(BinOp::Shr),
            Some(Token::PipeEq) => Some(BinOp::BitOr),
            Some(Token::AmpEq) => Some(BinOp::BitAnd),
            Some(Token::CaretEq) => Some(BinOp::BitXor),
            _ => None,
        };

        if let Some(op) = compound_op {
            self.advance();
            let rhs = self.parse_expr()?;
            // Desugar: lhs op= rhs  ->  lhs = lhs op rhs
            let binary = Expr::Binary {
                left: Box::new(lhs.clone()),
                op,
                right: Box::new(rhs),
            };
            return Ok(Expr::Assign {
                target: Box::new(lhs),
                value: Box::new(binary),
            });
        }

        // Check for Legion compound operators: ⊕=, ∂=, ⫰=
        match self.current_token() {
            Some(Token::DirectSumEq) => {
                // Superposition: field∿ ⊕= pattern
                self.advance();
                let pattern = self.parse_expr()?;
                return Ok(Expr::LegionSuperposition {
                    field: Box::new(lhs),
                    pattern: Box::new(pattern),
                });
            }
            Some(Token::PartialEq_) => {
                // Decay: field∿ ∂= rate
                self.advance();
                let rate = self.parse_expr()?;
                return Ok(Expr::LegionDecay {
                    field: Box::new(lhs),
                    rate: Box::new(rate),
                });
            }
            Some(Token::InterfereEq) => {
                // Interference assign (rare, but supported)
                self.advance();
                let query = self.parse_expr()?;
                return Ok(Expr::LegionInterference {
                    query: Box::new(query),
                    field: Box::new(lhs),
                });
            }
            _ => {}
        }

        Ok(lhs)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> ParseResult<Expr> {
        let mut lhs = self.parse_prefix_expr()?;

        loop {
            // Skip comments between binary operators - allows line-continuation style:
            // let x = foo()
            //     // comment
            //     && bar()
            self.skip_comments();

            // Check for pipe operator - but only if followed by pipe operation token
            // Otherwise treat | as bitwise OR for Rust-style code
            if self.check(&Token::Pipe) && self.peek_looks_like_pipe_op() {
                lhs = self.parse_pipe_chain(lhs)?;
                // After pipe chain, check for postfix operators like ? and method calls
                lhs = self.parse_postfix_after_pipe(lhs)?;
                continue;
            }

            // Check for binary operators
            let op = match self.current_token() {
                // Bitwise OR - only reached if not a pipe operation
                Some(Token::Pipe) => BinOp::BitOr,
                Some(Token::OrOr) => BinOp::Or,
                Some(Token::AndAnd) => BinOp::And,
                Some(Token::EqEq) => BinOp::Eq,
                Some(Token::NotEq) => BinOp::Ne,
                Some(Token::Lt) => BinOp::Lt,
                Some(Token::LtEq) => BinOp::Le,
                Some(Token::Gt) => BinOp::Gt,
                Some(Token::GtEq) => BinOp::Ge,
                Some(Token::Plus) => BinOp::Add,
                Some(Token::Minus) => BinOp::Sub,
                Some(Token::Star) => BinOp::Mul,
                Some(Token::Slash) => BinOp::Div,
                Some(Token::Percent) => BinOp::Rem,
                Some(Token::StarStar) => BinOp::Pow,
                Some(Token::Amp) => BinOp::BitAnd,
                Some(Token::Caret) => BinOp::BitXor,
                Some(Token::Shl) => BinOp::Shl,
                Some(Token::Shr) => BinOp::Shr,
                Some(Token::PlusPlus) => BinOp::Concat,
                // Matrix multiplication
                Some(Token::At) => BinOp::MatMul,
                // Unicode bitwise operators
                Some(Token::BitwiseAndSymbol) => BinOp::BitAnd, // ⋏
                Some(Token::BitwiseOrSymbol) => BinOp::BitOr,   // ⋎
                // Logical/geometric algebra operators
                Some(Token::LogicAnd) => BinOp::And, // ∧ (wedge/outer product, parsed as And)
                // Tensor/array operators
                Some(Token::CircledDot) => BinOp::Hadamard, // ⊙ element-wise multiply
                Some(Token::Tensor) => BinOp::TensorProd,   // ⊗ tensor product
                Some(Token::Gpu) => BinOp::Convolve,        // ⊛ convolution/merge
                // Legion operators handled specially below
                Some(Token::Interfere)
                | Some(Token::Distribute)
                | Some(Token::Broadcast)
                | Some(Token::Gather)
                | Some(Token::Consensus)
                | Some(Token::ConfidenceHigh) => {
                    // Handle Legion operators specially
                    lhs = self.parse_legion_operator(lhs)?;
                    continue;
                }
                _ => {
                    // Check for range operators: .. and ..=
                    // Range has very low precedence (lower than all binary ops)
                    // Only parse range if min_bp is low enough (i.e., we're not inside
                    // a higher-precedence expression like i+1..)
                    // Range binding power is 0 - lower than any binary operator
                    if min_bp == 0 && (self.check(&Token::DotDot) || self.check(&Token::DotDotEq)) {
                        let inclusive = self.consume_if(&Token::DotDotEq);
                        if !inclusive {
                            self.advance(); // consume ..
                        }
                        // Parse end of range (optional for open ranges like `0..`)
                        let end = if self.check(&Token::Semi)
                            || self.check(&Token::RBrace)
                            || self.check(&Token::Comma)
                            || self.check(&Token::RParen)
                            || self.check(&Token::RBracket)
                            || self.check(&Token::LBrace)
                        {
                            None
                        } else {
                            Some(Box::new(self.parse_expr_bp(0)?))
                        };
                        lhs = Expr::Range {
                            start: Some(Box::new(lhs)),
                            end,
                            inclusive,
                        };
                        continue;
                    }
                    break;
                }
            };

            let (l_bp, r_bp) = infix_binding_power(op);
            if l_bp < min_bp {
                break;
            }

            self.advance();
            // Skip comments before RHS - allows:
            // a ||  // comment
            //   b
            self.skip_comments();
            let rhs = self.parse_expr_bp(r_bp)?;

            lhs = Expr::Binary {
                left: Box::new(lhs),
                op,
                right: Box::new(rhs),
            };
        }

        Ok(lhs)
    }

    fn parse_prefix_expr(&mut self) -> ParseResult<Expr> {
        match self.current_token() {
            Some(Token::Minus) => {
                self.advance();
                let expr = self.parse_prefix_expr()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
            Some(Token::Bang) => {
                self.advance();
                let expr = self.parse_prefix_expr()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            Some(Token::Star) => {
                self.advance();
                let expr = self.parse_prefix_expr()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Deref,
                    expr: Box::new(expr),
                })
            }
            // Double dereference: **expr
            Some(Token::StarStar) => {
                self.advance();
                let inner = self.parse_prefix_expr()?;
                // Desugar **x to *(*x)
                let first_deref = Expr::Unary {
                    op: UnaryOp::Deref,
                    expr: Box::new(inner),
                };
                Ok(Expr::Unary {
                    op: UnaryOp::Deref,
                    expr: Box::new(first_deref),
                })
            }
            Some(Token::Amp) => {
                self.advance();
                let op = if self.consume_if(&Token::Mut) {
                    UnaryOp::RefMut
                } else {
                    UnaryOp::Ref
                };
                let expr = self.parse_prefix_expr()?;
                Ok(Expr::Unary {
                    op,
                    expr: Box::new(expr),
                })
            }
            // Prefix evidentiality markers: ?expr, ~expr, ‽expr
            Some(Token::Question) => {
                self.advance();
                let expr = self.parse_prefix_expr()?;
                Ok(Expr::Evidential {
                    expr: Box::new(expr),
                    evidentiality: Evidentiality::Uncertain,
                })
            }
            Some(Token::Tilde) => {
                self.advance();
                let expr = self.parse_prefix_expr()?;
                Ok(Expr::Evidential {
                    expr: Box::new(expr),
                    evidentiality: Evidentiality::Reported,
                })
            }
            Some(Token::Interrobang) => {
                self.advance();
                let expr = self.parse_prefix_expr()?;
                Ok(Expr::Evidential {
                    expr: Box::new(expr),
                    evidentiality: Evidentiality::Paradox,
                })
            }
            // Move closure: move |params| body or move || body
            Some(Token::Move) => {
                self.advance();
                self.parse_pipe_closure_with_move(true)
            }
            // Pipe-style closure: |params| body or || body
            Some(Token::Pipe) | Some(Token::OrOr) => self.parse_pipe_closure_with_move(false),
            _ => self.parse_postfix_expr(),
        }
    }

    /// Parse a pipe-style closure: |params| body or || body or move |params| body
    fn parse_pipe_closure_with_move(&mut self, is_move: bool) -> ParseResult<Expr> {
        let params = if self.consume_if(&Token::OrOr) {
            // || body - no parameters
            Vec::new()
        } else {
            // |params| body
            self.expect(Token::Pipe)?;
            let mut params = Vec::new();
            if !self.check(&Token::Pipe) {
                loop {
                    // Parse parameter pattern (possibly with type)
                    let pattern = self.parse_pattern()?;
                    let ty = if self.consume_if(&Token::Colon) {
                        Some(self.parse_type()?)
                    } else {
                        None
                    };
                    params.push(ClosureParam { pattern, ty });
                    if !self.consume_if(&Token::Comma) {
                        break;
                    }
                    if self.check(&Token::Pipe) {
                        break;
                    }
                }
            }
            self.expect(Token::Pipe)?;
            params
        };

        // Optional return type annotation: |params| -> Type { body }
        let return_type = if self.consume_if(&Token::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_expr()?;
        Ok(Expr::Closure {
            params,
            return_type,
            body: Box::new(body),
            is_move,
        })
    }

    /// Parse macro tokens: collects all tokens inside matching delimiters
    fn parse_macro_tokens(&mut self) -> ParseResult<String> {
        // Determine delimiter type
        let (open, close) = match self.current_token() {
            Some(Token::LParen) => (Token::LParen, Token::RParen),
            Some(Token::LBracket) => (Token::LBracket, Token::RBracket),
            Some(Token::LBrace) => (Token::LBrace, Token::RBrace),
            _ => {
                return Err(ParseError::Custom(
                    "expected '(', '[', or '{' for macro invocation".to_string(),
                ))
            }
        };

        self.advance(); // consume opening delimiter
        let mut tokens = String::new();
        let mut depth = 1;

        while depth > 0 && !self.is_eof() {
            if self.check(&open) {
                depth += 1;
            } else if self.check(&close) {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }

            // Collect token text (approximate - this is a simplified approach)
            if let Some((token, span)) = &self.current {
                // Get the source slice for this token
                let token_str = match token {
                    Token::Ident(s) => s.clone(),
                    Token::IntLit(s) => s.clone(),
                    Token::FloatLit(s) => s.clone(),
                    Token::StringLit(s) => {
                        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
                    }
                    Token::CharLit(c) => format!("'{}'", c),
                    Token::Comma => ",".to_string(),
                    Token::Colon => ":".to_string(),
                    Token::ColonColon => "::".to_string(),
                    Token::Dot => ".".to_string(),
                    Token::DotDot => "..".to_string(),
                    Token::Semi => ";".to_string(),
                    Token::LParen => "(".to_string(),
                    Token::RParen => ")".to_string(),
                    Token::LBrace => "{".to_string(),
                    Token::RBrace => "}".to_string(),
                    Token::LBracket => "[".to_string(),
                    Token::RBracket => "]".to_string(),
                    Token::Lt => "<".to_string(),
                    Token::Gt => ">".to_string(),
                    Token::Eq => "=".to_string(),
                    Token::FatArrow => "=>".to_string(),
                    Token::Bang => "!".to_string(),
                    Token::Question => "?".to_string(),
                    Token::Amp => "&".to_string(),
                    Token::Pipe => "|".to_string(),
                    Token::Underscore => "_".to_string(),
                    Token::Plus => "+".to_string(),
                    Token::Minus => "-".to_string(),
                    Token::Star => "*".to_string(),
                    Token::Slash => "/".to_string(),
                    Token::Percent => "%".to_string(),
                    Token::EqEq => "==".to_string(),
                    Token::NotEq => "!=".to_string(),
                    Token::LtEq => "<=".to_string(),
                    Token::GtEq => ">=".to_string(),
                    Token::AndAnd => "&&".to_string(),
                    Token::OrOr => "||".to_string(),
                    Token::Arrow => "->".to_string(),
                    Token::Hash => "#".to_string(),
                    Token::At => "@".to_string(),
                    Token::Tilde => "~".to_string(),
                    // Keywords
                    Token::SelfLower => "self".to_string(),
                    Token::SelfUpper => "Self".to_string(),
                    Token::Let => "let".to_string(),
                    Token::Mut => "mut".to_string(),
                    Token::Fn => "fn".to_string(),
                    Token::If => "if".to_string(),
                    Token::Else => "else".to_string(),
                    Token::Match => "match".to_string(),
                    Token::For => "for".to_string(),
                    Token::While => "while".to_string(),
                    Token::Loop => "loop".to_string(),
                    Token::Break => "break".to_string(),
                    Token::Continue => "continue".to_string(),
                    Token::Return => "return".to_string(),
                    Token::Struct => "struct".to_string(),
                    Token::Enum => "enum".to_string(),
                    Token::Impl => "impl".to_string(),
                    Token::Trait => "trait".to_string(),
                    Token::Type => "type".to_string(),
                    Token::Pub => "pub".to_string(),
                    Token::Mod => "mod".to_string(),
                    Token::Use => "use".to_string(),
                    Token::As => "as".to_string(),
                    Token::In => "in".to_string(),
                    Token::True => "true".to_string(),
                    Token::False => "false".to_string(),
                    Token::Null => "null".to_string(),
                    Token::Const => "const".to_string(),
                    Token::Static => "static".to_string(),
                    Token::Async => "async".to_string(),
                    Token::Await => "await".to_string(),
                    Token::Move => "move".to_string(),
                    Token::Ref => "ref".to_string(),
                    Token::Where => "where".to_string(),
                    Token::Dyn => "dyn".to_string(),
                    Token::Super => "super".to_string(),
                    Token::Crate => "crate".to_string(),
                    _ => format!("{:?}", token),
                };
                // Don't add space before . :: ( [ { ) ] } , ;
                let suppress_space_before = matches!(
                    token,
                    Token::Dot
                        | Token::ColonColon
                        | Token::LParen
                        | Token::LBracket
                        | Token::LBrace
                        | Token::RParen
                        | Token::RBracket
                        | Token::RBrace
                        | Token::Comma
                        | Token::Semi
                );
                if !tokens.is_empty()
                    && !suppress_space_before
                    && !tokens.ends_with('.')
                    && !tokens.ends_with("::")
                    && !tokens.ends_with('(')
                    && !tokens.ends_with('[')
                    && !tokens.ends_with('{')
                {
                    tokens.push(' ');
                }
                tokens.push_str(&token_str);
            }
            self.advance();
        }

        self.expect(close)?;
        Ok(tokens)
    }

    /// Check if an expression is a block-ending expression that should not be callable.
    /// Block expressions like if/while/match/loop/for return values but should not
    /// be directly called like functions.
    fn is_non_callable_expr(expr: &Expr) -> bool {
        matches!(
            expr,
            Expr::If { .. }
                | Expr::While { .. }
                | Expr::Match { .. }
                | Expr::Loop { .. }
                | Expr::For { .. }
                | Expr::Block(_)
        )
    }

    fn parse_postfix_expr(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            // Skip comments between postfix operations - allows line-continuation:
            // foo()
            //     // comment
            //     .bar()
            self.skip_comments();

            match self.current_token() {
                Some(Token::LParen) => {
                    // Don't treat block-ending expressions as callable
                    // This prevents parsing `if {...} (...)` as a call expression
                    if Self::is_non_callable_expr(&expr) {
                        break;
                    }
                    self.advance();
                    let args = self.parse_expr_list()?;
                    self.expect(Token::RParen)?;
                    expr = Expr::Call {
                        func: Box::new(expr),
                        args,
                    };
                }
                Some(Token::LBracket) => {
                    self.advance();
                    // Support multi-dimensional indexing: arr[.., 0] or arr[i, j, k]
                    let first = self.parse_expr()?;
                    if self.consume_if(&Token::Comma) {
                        // Multi-dimensional index: create tuple of indices
                        let mut indices = vec![first];
                        while !self.check(&Token::RBracket) && !self.is_eof() {
                            indices.push(self.parse_expr()?);
                            if !self.consume_if(&Token::Comma) {
                                break;
                            }
                        }
                        self.expect(Token::RBracket)?;
                        // Represent as index with a tuple expression
                        expr = Expr::Index {
                            expr: Box::new(expr),
                            index: Box::new(Expr::Tuple(indices)),
                        };
                    } else {
                        self.expect(Token::RBracket)?;
                        expr = Expr::Index {
                            expr: Box::new(expr),
                            index: Box::new(first),
                        };
                    }
                }
                Some(Token::Dot) => {
                    self.advance();
                    // Handle `.⌛` as await syntax (alternative to `expr⌛`)
                    if self.check(&Token::Hourglass) {
                        self.advance();
                        let evidentiality = self.parse_evidentiality_opt();
                        expr = Expr::Await {
                            expr: Box::new(expr),
                            evidentiality,
                        };
                        continue;
                    }
                    // Handle both named fields (`.field`) and tuple indices (`.0`, `.1`)
                    let field = if let Some(Token::IntLit(idx)) = self.current_token() {
                        let idx = idx.clone();
                        let span = self.current_span();
                        self.advance();
                        Ident {
                            name: idx,
                            evidentiality: None,
                            affect: None,
                            span,
                        }
                    } else {
                        self.parse_ident()?
                    };
                    // Check for turbofish syntax: method::<Type>(args)
                    if self.check(&Token::ColonColon) {
                        self.advance(); // consume ::
                        self.expect(Token::Lt)?;
                        // Temporarily exit condition context - turbofish is type context
                        let was_in_condition = self.in_condition;
                        self.in_condition = false;
                        let type_args = self.parse_type_list()?;
                        self.expect_gt()?;
                        self.in_condition = was_in_condition;
                        self.expect(Token::LParen)?;
                        let args = self.parse_expr_list()?;
                        self.expect(Token::RParen)?;
                        expr = Expr::MethodCall {
                            receiver: Box::new(expr),
                            method: field,
                            type_args: Some(type_args),
                            args,
                        };
                    } else if self.check(&Token::LParen) {
                        self.advance();
                        let args = self.parse_expr_list()?;
                        self.expect(Token::RParen)?;
                        expr = Expr::MethodCall {
                            receiver: Box::new(expr),
                            method: field,
                            type_args: None,
                            args,
                        };
                    } else {
                        // Consume optional unambiguous evidentiality markers after field: self.field◊, self.field~
                        // Note: don't consume ! or ? here as they have other meanings (! = not, ? = try)
                        while self.check(&Token::Tilde) || self.check(&Token::Lozenge) {
                            self.advance();
                        }
                        expr = Expr::Field {
                            expr: Box::new(expr),
                            field,
                        };
                    }
                }
                Some(Token::Question) => {
                    // Check if this is Type? { ... } struct literal with evidentiality
                    // vs expr? try operator
                    if self.peek_next() == Some(&Token::LBrace) && !self.is_in_condition() {
                        if let Expr::Path(ref path) = expr {
                            let path = path.clone();
                            self.advance(); // consume ?
                            self.advance(); // consume {
                            let (fields, rest) = self.parse_struct_fields()?;
                            self.expect(Token::RBrace)?;
                            expr = Expr::Struct { path, fields, rest };
                            continue;
                        }
                    }
                    // Not a struct literal - treat as try operator
                    self.advance();
                    expr = Expr::Try(Box::new(expr));
                }
                // Cast expression: expr as Type
                Some(Token::As) => {
                    self.advance();
                    let ty = self.parse_type()?;
                    expr = Expr::Cast {
                        expr: Box::new(expr),
                        ty,
                    };
                }
                Some(Token::Bang) => {
                    // Check for macro invocation: path!(...)  path![...]  path!{...}
                    if let Expr::Path(path) = &expr {
                        let peeked = self.peek_next();
                        // Peek at next token to see if it's a macro delimiter
                        let is_macro = match peeked {
                            Some(Token::LParen) | Some(Token::LBracket) | Some(Token::LBrace) => {
                                true
                            }
                            _ => false,
                        };
                        if is_macro {
                            self.advance(); // consume !
                            let tokens = self.parse_macro_tokens()?;
                            expr = Expr::Macro {
                                path: path.clone(),
                                tokens,
                            };
                            continue;
                        }
                    }
                    // Not a macro, check for evidentiality
                    if let Some(ev) = self.parse_evidentiality_opt() {
                        expr = Expr::Evidential {
                            expr: Box::new(expr),
                            evidentiality: ev,
                        };
                    } else {
                        break;
                    }
                }
                Some(Token::Tilde) | Some(Token::Interrobang) | Some(Token::Lozenge) => {
                    if let Some(ev) = self.parse_evidentiality_opt() {
                        // After evidentiality marker, check for struct literal: Type~ { ... }
                        // The evidentiality attaches to the struct type
                        if self.check(&Token::LBrace) && !self.is_in_condition() {
                            if let Expr::Path(ref path) = expr {
                                let path = path.clone();
                                self.advance(); // consume {
                                let (fields, rest) = self.parse_struct_fields()?;
                                self.expect(Token::RBrace)?;
                                expr = Expr::Struct { path, fields, rest };
                                continue;
                            }
                        }
                        // Wrap in Evidential for non-struct cases
                        expr = Expr::Evidential {
                            expr: Box::new(expr),
                            evidentiality: ev,
                        };
                    } else {
                        break;
                    }
                }
                Some(Token::ColonColon) => {
                    // Qualified path call: Type::method() or path!::method()
                    self.advance(); // consume ::
                    let method = self.parse_ident()?;
                    // Check for turbofish: Type::method::<T>()
                    let type_args = if self.check(&Token::ColonColon) {
                        self.advance();
                        self.expect(Token::Lt)?;
                        let types = self.parse_type_list()?;
                        self.expect_gt()?;
                        Some(types)
                    } else {
                        None
                    };
                    if self.check(&Token::LParen) {
                        self.advance();
                        let args = self.parse_expr_list()?;
                        self.expect(Token::RParen)?;
                        // Treat as a static method call - desugar to path::method(args)
                        expr = Expr::MethodCall {
                            receiver: Box::new(expr),
                            method,
                            type_args,
                            args,
                        };
                    } else {
                        // Field access style: Type::CONST
                        expr = Expr::Field {
                            expr: Box::new(expr),
                            field: method,
                        };
                    }
                }
                Some(Token::Hourglass) => {
                    self.advance();
                    // Check for optional evidentiality marker: ⌛? ⌛! ⌛~ ⌛◊ ⌛‽
                    let evidentiality = match self.current_token() {
                        Some(Token::Question) => {
                            self.advance();
                            Some(Evidentiality::Uncertain)
                        }
                        Some(Token::Bang) => {
                            self.advance();
                            Some(Evidentiality::Known)
                        }
                        Some(Token::Tilde) => {
                            self.advance();
                            Some(Evidentiality::Reported)
                        }
                        Some(Token::Lozenge) => {
                            self.advance();
                            Some(Evidentiality::Predicted)
                        }
                        Some(Token::Interrobang) => {
                            self.advance();
                            Some(Evidentiality::Paradox)
                        }
                        _ => None,
                    };
                    expr = Expr::Await {
                        expr: Box::new(expr),
                        evidentiality,
                    };
                }
                // Incorporation: expr·verb·noun·action
                // Polysynthetic noun incorporation using middle dot
                Some(Token::MiddleDot) => {
                    // Convert current expr to first segment, then parse chain
                    let first_segment = self.expr_to_incorporation_segment(expr.clone())?;
                    let mut segments = vec![first_segment];

                    while self.consume_if(&Token::MiddleDot) {
                        // Handle both named methods and tuple indices: ·method() or ·0
                        let name = if let Some(Token::IntLit(idx)) = self.current_token().cloned() {
                            let span = self.current_span();
                            self.advance();
                            Ident {
                                name: idx,
                                evidentiality: None,
                                affect: None,
                                span,
                            }
                        } else {
                            self.parse_ident()?
                        };
                        // Handle bracket generics on method: ·method[Type]()
                        if self.check(&Token::LBracket) && self.peek_looks_like_bracket_generic() {
                            self.advance(); // consume [
                            let _generics = self.parse_type_list()?;
                            self.expect(Token::RBracket)?;
                            // Generics are parsed but currently ignored in incorporation segments
                        }
                        // Handle angle bracket generics: ·method<Type>()
                        if self.check(&Token::Lt) && self.peek_looks_like_generic_arg() {
                            self.advance(); // consume <
                            let _generics = self.parse_type_list()?;
                            self.expect_gt()?;
                        }
                        let args = if self.check(&Token::LParen) {
                            self.advance();
                            let args = self.parse_expr_list()?;
                            self.expect(Token::RParen)?;
                            Some(args)
                        } else {
                            None
                        };
                        segments.push(IncorporationSegment { name, args });
                    }

                    expr = Expr::Incorporation { segments };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary_expr(&mut self) -> ParseResult<Expr> {
        match self.current_token().cloned() {
            Some(Token::IntLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Decimal,
                    suffix: None,
                }))
            }
            Some(Token::BinaryLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Binary,
                    suffix: None,
                }))
            }
            Some(Token::OctalLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Octal,
                    suffix: None,
                }))
            }
            Some(Token::HexLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Hex,
                    suffix: None,
                }))
            }
            Some(Token::VigesimalLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Vigesimal,
                    suffix: None,
                }))
            }
            Some(Token::SexagesimalLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Sexagesimal,
                    suffix: None,
                }))
            }
            Some(Token::DuodecimalLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Int {
                    value: s,
                    base: NumBase::Duodecimal,
                    suffix: None,
                }))
            }
            Some(Token::FloatLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Float {
                    value: s,
                    suffix: None,
                }))
            }
            Some(Token::StringLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::String(s)))
            }
            Some(Token::MultiLineStringLit(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::MultiLineString(s)))
            }
            Some(Token::RawStringLit(s)) | Some(Token::RawStringDelimited(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::RawString(s)))
            }
            // Full range expression: `..` or `..=` (with optional end)
            // Handles cases like `.drain(..)` and `[..]`
            Some(Token::DotDot) | Some(Token::DotDotEq) => {
                let inclusive = self.consume_if(&Token::DotDotEq);
                if !inclusive {
                    self.advance(); // consume ..
                }
                // Check if there's an end expression
                let end = if self.check(&Token::RParen)
                    || self.check(&Token::RBracket)
                    || self.check(&Token::Comma)
                    || self.check(&Token::Semi)
                    || self.check(&Token::RBrace)
                {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                Ok(Expr::Range {
                    start: None,
                    end,
                    inclusive,
                })
            }
            Some(Token::ByteStringLit(bytes)) => {
                self.advance();
                Ok(Expr::Literal(Literal::ByteString(bytes)))
            }
            Some(Token::InterpolatedStringLit(s)) => {
                self.advance();
                // Parse the interpolation parts
                let parts = self.parse_interpolation_parts(&s)?;
                Ok(Expr::Literal(Literal::InterpolatedString { parts }))
            }
            Some(Token::SigilStringSql(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::SigilStringSql(s)))
            }
            Some(Token::SigilStringRoute(s)) => {
                self.advance();
                Ok(Expr::Literal(Literal::SigilStringRoute(s)))
            }
            Some(Token::CharLit(c)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Char(c)))
            }
            Some(Token::ByteCharLit(b)) => {
                self.advance();
                Ok(Expr::Literal(Literal::ByteChar(b)))
            }
            Some(Token::True) => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(true)))
            }
            Some(Token::False) => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(false)))
            }
            Some(Token::Null) => {
                self.advance();
                Ok(Expr::Literal(Literal::Null))
            }
            Some(Token::Empty) => {
                self.advance();
                Ok(Expr::Literal(Literal::Empty))
            }
            Some(Token::Infinity) => {
                self.advance();
                Ok(Expr::Literal(Literal::Infinity))
            }
            Some(Token::Circle) => {
                self.advance();
                Ok(Expr::Literal(Literal::Circle))
            }
            Some(Token::LParen) => {
                self.advance();
                if self.check(&Token::RParen) {
                    self.advance();
                    return Ok(Expr::Tuple(vec![]));
                }
                let expr = self.parse_expr()?;
                if self.consume_if(&Token::Comma) {
                    let mut exprs = vec![expr];
                    while !self.check(&Token::RParen) {
                        exprs.push(self.parse_expr()?);
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Expr::Tuple(exprs))
                } else {
                    self.expect(Token::RParen)?;
                    Ok(expr)
                }
            }
            Some(Token::LBracket) => {
                self.advance();
                // Check for empty array
                if self.check(&Token::RBracket) {
                    self.advance();
                    return Ok(Expr::Array(vec![]));
                }
                // Parse first expression
                let first = self.parse_expr()?;
                // Check for repeat syntax: [value; count]
                if self.consume_if(&Token::Semi) {
                    let count = self.parse_expr()?;
                    self.expect(Token::RBracket)?;
                    return Ok(Expr::ArrayRepeat {
                        value: Box::new(first),
                        count: Box::new(count),
                    });
                }
                // Otherwise, parse as regular array literal
                let mut exprs = vec![first];
                while self.consume_if(&Token::Comma) {
                    // Skip comments after comma (for trailing comments)
                    self.skip_comments();
                    if self.check(&Token::RBracket) {
                        break; // trailing comma
                    }
                    exprs.push(self.parse_expr()?);
                }
                self.skip_comments();
                self.expect(Token::RBracket)?;
                Ok(Expr::Array(exprs))
            }
            Some(Token::LBrace) => {
                // Could be block or closure
                self.parse_block_or_closure()
            }
            Some(Token::If) => self.parse_if_expr(),
            Some(Token::Match) => self.parse_match_expr(),
            Some(Token::Unsafe) => {
                self.advance();
                let block = self.parse_block()?;
                Ok(Expr::Unsafe(block))
            }
            Some(Token::Async) => {
                self.advance();
                let is_move = self.consume_if(&Token::Move);
                let block = self.parse_block()?;
                Ok(Expr::Async { block, is_move })
            }
            Some(Token::Const) => {
                // Const block expression: `const { expr }` - compile-time evaluated block
                // For now, parse as a regular block expression
                self.advance();
                let block = self.parse_block()?;
                Ok(Expr::Block(block))
            }
            Some(Token::Lifetime(name)) => {
                // Labeled loop: 'label: loop/while/for { ... }
                let span = self.current_span();
                let label = Ident {
                    name: name.clone(),
                    evidentiality: None,
                    affect: None,
                    span,
                };
                self.advance();
                self.expect(Token::Colon)?;
                match self.current_token().cloned() {
                    Some(Token::Loop) => {
                        self.advance();
                        let body = self.parse_block()?;
                        Ok(Expr::Loop {
                            label: Some(label),
                            body,
                        })
                    }
                    Some(Token::While) => {
                        self.advance();
                        // Handle while-let: `while let pattern = expr { ... }`
                        let condition = if self.consume_if(&Token::Let) {
                            let pattern = self.parse_pattern()?;
                            self.expect(Token::Eq)?;
                            let value = self.parse_condition()?;
                            Expr::Let {
                                pattern,
                                value: Box::new(value),
                            }
                        } else {
                            self.parse_condition()?
                        };
                        let body = self.parse_block()?;
                        Ok(Expr::While {
                            label: Some(label),
                            condition: Box::new(condition),
                            body,
                        })
                    }
                    Some(Token::For) => {
                        self.advance();
                        let pattern = self.parse_pattern()?;
                        self.expect(Token::In)?;
                        let iter = self.parse_condition()?;
                        let body = self.parse_block()?;
                        Ok(Expr::For {
                            label: Some(label),
                            pattern,
                            iter: Box::new(iter),
                            body,
                        })
                    }
                    other => Err(ParseError::UnexpectedToken {
                        expected: "loop, while, or for after label".to_string(),
                        found: other.unwrap_or(Token::Null),
                        span: self.current_span(),
                    }),
                }
            }
            Some(Token::Loop) => {
                self.advance();
                let body = self.parse_block()?;
                Ok(Expr::Loop { label: None, body })
            }
            Some(Token::While) => {
                self.advance();
                // Handle while-let: `while let pattern = expr { ... }`
                let condition = if self.consume_if(&Token::Let) {
                    let pattern = self.parse_pattern()?;
                    self.expect(Token::Eq)?;
                    let value = self.parse_condition()?;
                    Expr::Let {
                        pattern,
                        value: Box::new(value),
                    }
                } else {
                    self.parse_condition()?
                };
                let body = self.parse_block()?;
                Ok(Expr::While {
                    label: None,
                    condition: Box::new(condition),
                    body,
                })
            }
            Some(Token::For) => {
                self.advance();
                let pattern = self.parse_pattern()?;
                self.expect(Token::In)?;
                let iter = self.parse_condition()?;
                let body = self.parse_block()?;
                Ok(Expr::For {
                    label: None,
                    pattern,
                    iter: Box::new(iter),
                    body,
                })
            }
            Some(Token::Return) => {
                self.advance();
                // Check for terminators: ; } or , (in match arms)
                let value = if self.check(&Token::Semi)
                    || self.check(&Token::RBrace)
                    || self.check(&Token::Comma)
                {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                Ok(Expr::Return(value))
            }
            Some(Token::Break) => {
                self.advance();
                // Check for optional label: break 'label or break 'label value
                let label = if let Some(Token::Lifetime(name)) = self.current_token().cloned() {
                    let span = self.current_span();
                    let label = Ident {
                        name,
                        evidentiality: None,
                        affect: None,
                        span,
                    };
                    self.advance();
                    Some(label)
                } else {
                    None
                };
                // Check for terminators: ; } or , (in match arms)
                let value = if self.check(&Token::Semi)
                    || self.check(&Token::RBrace)
                    || self.check(&Token::Comma)
                {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                Ok(Expr::Break { label, value })
            }
            Some(Token::Continue) => {
                self.advance();
                // Check for optional label: continue 'label
                let label = if let Some(Token::Lifetime(name)) = self.current_token().cloned() {
                    let span = self.current_span();
                    let label = Ident {
                        name,
                        evidentiality: None,
                        affect: None,
                        span,
                    };
                    self.advance();
                    Some(label)
                } else {
                    None
                };
                Ok(Expr::Continue { label })
            }
            // Morphemes as standalone expressions
            Some(Token::Tau) | Some(Token::Phi) | Some(Token::Sigma) | Some(Token::Rho)
            | Some(Token::Lambda) | Some(Token::Pi) => {
                let kind = self.parse_morpheme_kind()?;
                if self.check(&Token::LBrace) {
                    self.advance();
                    self.skip_comments();
                    // Check for closure pattern: τ{x => expr} or τ{(a, b) => expr}
                    let body = if self.looks_like_morpheme_closure() {
                        self.parse_morpheme_closure()?
                    } else {
                        self.parse_expr()?
                    };
                    self.expect(Token::RBrace)?;
                    Ok(Expr::Morpheme {
                        kind,
                        body: Box::new(body),
                    })
                } else {
                    // Just the morpheme symbol
                    Ok(Expr::Morpheme {
                        kind,
                        body: Box::new(Expr::Path(TypePath {
                            segments: vec![PathSegment {
                                ident: Ident {
                                    name: "_".to_string(),
                                    evidentiality: None,
                                    affect: None,
                                    span: Span::default(),
                                },
                                generics: None,
                            }],
                        })),
                    })
                }
            }
            // Sacred constants as expressions
            Some(Token::Sqrt) => {
                // Handle √2, √3, √5, etc. (square root constants)
                let span = self.current_span();
                self.advance();
                let name = if let Some(Token::IntLit(n)) = self.current_token().cloned() {
                    let merged_span = span.merge(self.current_span());
                    self.advance();
                    (format!("√{}", n), merged_span)
                } else {
                    ("√".to_string(), span)
                };
                Ok(Expr::Path(TypePath {
                    segments: vec![PathSegment {
                        ident: Ident {
                            name: name.0,
                            evidentiality: None,
                            affect: None,
                            span: name.1,
                        },
                        generics: None,
                    }],
                }))
            }
            Some(Token::Underscore) => {
                // Underscore as placeholder in closures
                let span = self.current_span();
                self.advance();
                Ok(Expr::Path(TypePath {
                    segments: vec![PathSegment {
                        ident: Ident {
                            name: "_".to_string(),
                            evidentiality: None,
                            affect: None,
                            span,
                        },
                        generics: None,
                    }],
                }))
            }
            Some(Token::SelfLower) => {
                // self keyword as expression
                let span = self.current_span();
                self.advance();
                Ok(Expr::Path(TypePath {
                    segments: vec![PathSegment {
                        ident: Ident {
                            name: "self".to_string(),
                            evidentiality: None,
                            affect: None,
                            span,
                        },
                        generics: None,
                    }],
                }))
            }
            Some(Token::SelfUpper) => {
                // Self keyword as expression (struct constructor or path start)
                let span = self.current_span();
                self.advance();
                let mut segments = vec![PathSegment {
                    ident: Ident {
                        name: "Self".to_string(),
                        evidentiality: None,
                        affect: None,
                        span,
                    },
                    generics: None,
                }];
                // Handle Self::method() and other path continuations
                while self.consume_if(&Token::ColonColon) {
                    // Check for turbofish syntax: Self::method::<Type>
                    if self.check(&Token::Lt) {
                        self.advance(); // consume <
                        let types = self.parse_type_list()?;
                        self.expect_gt()?;
                        // Update the last segment with these generics
                        if let Some(last) = segments.last_mut() {
                            last.generics = Some(types);
                        }
                        break;
                    }
                    segments.push(self.parse_path_segment()?);
                }
                let path = TypePath { segments };
                // Check for struct literal: Self { ... }
                if self.check(&Token::LBrace) && !self.is_in_condition() {
                    self.advance();
                    let (fields, rest) = self.parse_struct_fields()?;
                    self.expect(Token::RBrace)?;
                    Ok(Expr::Struct { path, fields, rest })
                } else {
                    Ok(Expr::Path(path))
                }
            }
            Some(Token::Ident(_)) => {
                let path = self.parse_type_path()?;

                // Check for struct literal: Name { ... }
                // Note: Name! { ... } is treated as a macro invocation, handled in parse_postfix_expr
                // This allows html! { ... } and other macros with brace bodies to work correctly
                if self.check(&Token::LBrace) && !self.is_in_condition() {
                    self.advance();
                    let (fields, rest) = self.parse_struct_fields()?;
                    self.expect(Token::RBrace)?;
                    Ok(Expr::Struct { path, fields, rest })
                } else {
                    Ok(Expr::Path(path))
                }
            }
            Some(Token::Asm) => self.parse_inline_asm(),
            Some(Token::Volatile) => self.parse_volatile_expr(),
            Some(Token::Simd) => self.parse_simd_expr(),
            Some(Token::Atomic) => self.parse_atomic_expr(),
            // Implicit self field access: `.field` desugars to `self.field`
            // This allows more concise method bodies:
            //   fn increment(mut self) { .count += 1; }
            // instead of:
            //   fn increment(mut self) { self.count += 1; }
            Some(Token::Dot) => {
                let dot_span = self.current_span();
                self.advance(); // consume .
                match self.current_token() {
                    Some(Token::Ident(name)) => {
                        let field_name = name.clone();
                        let field_span = self.current_span();
                        self.advance();
                        // Create `self.field` expression
                        let self_expr = Expr::Path(TypePath {
                            segments: vec![PathSegment {
                                ident: Ident {
                                    name: "self".to_string(),
                                    evidentiality: None,
                                    affect: None,
                                    span: dot_span,
                                },
                                generics: None,
                            }],
                        });
                        Ok(Expr::Field {
                            expr: Box::new(self_expr),
                            field: Ident {
                                name: field_name,
                                evidentiality: None,
                                affect: None,
                                span: field_span,
                            },
                        })
                    }
                    Some(token) => Err(ParseError::UnexpectedToken {
                        expected: "identifier after '.' for implicit self field".to_string(),
                        found: token.clone(),
                        span: self.current_span(),
                    }),
                    None => Err(ParseError::UnexpectedEof),
                }
            }
            // Handle contextual keywords as identifiers in expressions
            Some(ref token) if Self::keyword_as_ident(token).is_some() => {
                let path = self.parse_type_path()?;
                // Check for struct literal: Name { ... }
                // Note: Don't consume ! here - macro invocations (Name!(...)) are handled in parse_postfix_expr
                if self.check(&Token::LBrace) && !self.is_in_condition() {
                    self.advance();
                    let (fields, rest) = self.parse_struct_fields()?;
                    self.expect(Token::RBrace)?;
                    Ok(Expr::Struct { path, fields, rest })
                } else {
                    Ok(Expr::Path(path))
                }
            }
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: token,
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    /// Parse inline assembly: `asm!("template", ...)`
    ///
    /// Syntax:
    /// ```sigil
    /// asm!("template {0} {1}",
    ///     out("rax") result,
    ///     in("rbx") input,
    ///     clobber("rcx", "rdx"),
    ///     options(volatile, nostack))
    /// ```
    fn parse_inline_asm(&mut self) -> ParseResult<Expr> {
        self.expect(Token::Asm)?;
        self.expect(Token::Bang)?;
        self.expect(Token::LParen)?;

        // Parse template string
        let template = match self.current_token().cloned() {
            Some(Token::StringLit(s)) => {
                self.advance();
                s
            }
            Some(t) => {
                return Err(ParseError::UnexpectedToken {
                    expected: "assembly template string".to_string(),
                    found: t,
                    span: self.current_span(),
                });
            }
            None => return Err(ParseError::UnexpectedEof),
        };

        let mut outputs = Vec::new();
        let mut inputs = Vec::new();
        let mut clobbers = Vec::new();
        let mut options = AsmOptions::default();

        // Parse operands and options
        while self.consume_if(&Token::Comma) {
            if self.check(&Token::RParen) {
                break;
            }

            match self.current_token().cloned() {
                Some(Token::Ident(ref name)) if name == "out" => {
                    self.advance();
                    let operand = self.parse_asm_operand(AsmOperandKind::Output)?;
                    outputs.push(operand);
                }
                // Handle `in` which is a keyword (Token::In)
                Some(Token::In) => {
                    self.advance();
                    let operand = self.parse_asm_operand(AsmOperandKind::Input)?;
                    inputs.push(operand);
                }
                Some(Token::Ident(ref name)) if name == "inout" => {
                    self.advance();
                    let operand = self.parse_asm_operand(AsmOperandKind::InOut)?;
                    outputs.push(operand);
                }
                Some(Token::Ident(ref name)) if name == "clobber" => {
                    self.advance();
                    self.expect(Token::LParen)?;
                    while !self.check(&Token::RParen) {
                        if let Some(Token::StringLit(reg)) = self.current_token().cloned() {
                            self.advance();
                            clobbers.push(reg);
                        } else if let Some(Token::Ident(reg)) = self.current_token().cloned() {
                            self.advance();
                            clobbers.push(reg);
                        }
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RParen)?;
                }
                Some(Token::Ident(ref name)) if name == "options" => {
                    self.advance();
                    self.expect(Token::LParen)?;
                    while !self.check(&Token::RParen) {
                        if let Some(Token::Ident(opt)) = self.current_token().cloned() {
                            self.advance();
                            match opt.as_str() {
                                "volatile" => options.volatile = true,
                                "nostack" => options.nostack = true,
                                "pure" => options.pure_asm = true,
                                "readonly" => options.readonly = true,
                                "nomem" => options.nomem = true,
                                "att_syntax" => options.att_syntax = true,
                                _ => {}
                            }
                        }
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RParen)?;
                }
                _ => break,
            }
        }

        self.expect(Token::RParen)?;

        Ok(Expr::InlineAsm(InlineAsm {
            template,
            outputs,
            inputs,
            clobbers,
            options,
        }))
    }

    /// Parse an assembly operand: `("reg") expr` or `("reg") var => expr`
    fn parse_asm_operand(&mut self, kind: AsmOperandKind) -> ParseResult<AsmOperand> {
        self.expect(Token::LParen)?;

        let constraint = match self.current_token().cloned() {
            Some(Token::StringLit(s)) => {
                self.advance();
                s
            }
            Some(Token::Ident(s)) => {
                self.advance();
                s
            }
            Some(t) => {
                return Err(ParseError::UnexpectedToken {
                    expected: "register constraint".to_string(),
                    found: t,
                    span: self.current_span(),
                });
            }
            None => return Err(ParseError::UnexpectedEof),
        };

        self.expect(Token::RParen)?;

        let expr = self.parse_expr()?;

        // For inout, check for `=> output`
        let output = if kind == AsmOperandKind::InOut && self.consume_if(&Token::FatArrow) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        Ok(AsmOperand {
            constraint,
            expr,
            kind,
            output,
        })
    }

    /// Parse volatile memory operations
    ///
    /// - `volatile read<T>(ptr)` - volatile read from pointer
    /// - `volatile write<T>(ptr, value)` - volatile write to pointer
    fn parse_volatile_expr(&mut self) -> ParseResult<Expr> {
        self.expect(Token::Volatile)?;

        match self.current_token().cloned() {
            Some(Token::Ident(ref name)) if name == "read" => {
                self.advance();

                // Optional type parameter <T>
                let ty = if self.consume_if(&Token::Lt) {
                    let t = self.parse_type()?;
                    self.expect_gt()?;
                    Some(t)
                } else {
                    None
                };

                self.expect(Token::LParen)?;
                let ptr = self.parse_expr()?;
                self.expect(Token::RParen)?;

                Ok(Expr::VolatileRead {
                    ptr: Box::new(ptr),
                    ty,
                })
            }
            Some(Token::Ident(ref name)) if name == "write" => {
                self.advance();

                // Optional type parameter <T>
                let ty = if self.consume_if(&Token::Lt) {
                    let t = self.parse_type()?;
                    self.expect_gt()?;
                    Some(t)
                } else {
                    None
                };

                self.expect(Token::LParen)?;
                let ptr = self.parse_expr()?;
                self.expect(Token::Comma)?;
                let value = self.parse_expr()?;
                self.expect(Token::RParen)?;

                Ok(Expr::VolatileWrite {
                    ptr: Box::new(ptr),
                    value: Box::new(value),
                    ty,
                })
            }
            Some(t) => Err(ParseError::UnexpectedToken {
                expected: "'read' or 'write' after 'volatile'".to_string(),
                found: t,
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    /// Parse SIMD expressions
    ///
    /// Syntax:
    /// ```sigil
    /// simd[1.0, 2.0, 3.0, 4.0]              // SIMD literal
    /// simd.splat(1.0, 4)                    // Broadcast value to all lanes
    /// simd.add(a, b)                        // SIMD intrinsic
    /// simd.shuffle(a, b, [0, 4, 1, 5])      // Shuffle lanes
    /// simd.extract(v, 0)                    // Extract element
    /// simd.insert(v, 0, val)                // Insert element
    /// ```
    fn parse_simd_expr(&mut self) -> ParseResult<Expr> {
        self.expect(Token::Simd)?;

        match self.current_token().cloned() {
            Some(Token::LBracket) => {
                // SIMD literal: simd[1.0, 2.0, 3.0, 4.0]
                self.advance();
                let elements = self.parse_expr_list()?;
                self.expect(Token::RBracket)?;

                // Optional type annotation
                let ty = if self.consume_if(&Token::Colon) {
                    Some(self.parse_type()?)
                } else {
                    None
                };

                Ok(Expr::SimdLiteral { elements, ty })
            }
            Some(Token::Dot) => {
                self.advance();
                match self.current_token().cloned() {
                    Some(Token::Ident(ref op)) => {
                        let op_name = op.clone();
                        self.advance();
                        self.expect(Token::LParen)?;

                        match op_name.as_str() {
                            "splat" => {
                                let value = self.parse_expr()?;
                                self.expect(Token::Comma)?;
                                let lanes = match self.current_token() {
                                    Some(Token::IntLit(s)) => {
                                        let n = s.parse::<u8>().map_err(|_| {
                                            ParseError::Custom("invalid lane count".to_string())
                                        })?;
                                        self.advance();
                                        n
                                    }
                                    _ => {
                                        return Err(ParseError::Custom(
                                            "expected lane count".to_string(),
                                        ))
                                    }
                                };
                                self.expect(Token::RParen)?;
                                Ok(Expr::SimdSplat {
                                    value: Box::new(value),
                                    lanes,
                                })
                            }
                            "shuffle" => {
                                let a = self.parse_expr()?;
                                self.expect(Token::Comma)?;
                                let b = self.parse_expr()?;
                                self.expect(Token::Comma)?;
                                self.expect(Token::LBracket)?;
                                let mut indices = Vec::new();
                                loop {
                                    match self.current_token() {
                                        Some(Token::IntLit(s)) => {
                                            let n = s.parse::<u8>().map_err(|_| {
                                                ParseError::Custom("invalid index".to_string())
                                            })?;
                                            indices.push(n);
                                            self.advance();
                                        }
                                        _ => {
                                            return Err(ParseError::Custom(
                                                "expected index".to_string(),
                                            ))
                                        }
                                    }
                                    if !self.consume_if(&Token::Comma) {
                                        break;
                                    }
                                }
                                self.expect(Token::RBracket)?;
                                self.expect(Token::RParen)?;
                                Ok(Expr::SimdShuffle {
                                    a: Box::new(a),
                                    b: Box::new(b),
                                    indices,
                                })
                            }
                            "extract" => {
                                let vector = self.parse_expr()?;
                                self.expect(Token::Comma)?;
                                let index = match self.current_token() {
                                    Some(Token::IntLit(s)) => {
                                        let n = s.parse::<u8>().map_err(|_| {
                                            ParseError::Custom("invalid index".to_string())
                                        })?;
                                        self.advance();
                                        n
                                    }
                                    _ => {
                                        return Err(ParseError::Custom(
                                            "expected index".to_string(),
                                        ))
                                    }
                                };
                                self.expect(Token::RParen)?;
                                Ok(Expr::SimdExtract {
                                    vector: Box::new(vector),
                                    index,
                                })
                            }
                            "insert" => {
                                let vector = self.parse_expr()?;
                                self.expect(Token::Comma)?;
                                let index = match self.current_token() {
                                    Some(Token::IntLit(s)) => {
                                        let n = s.parse::<u8>().map_err(|_| {
                                            ParseError::Custom("invalid index".to_string())
                                        })?;
                                        self.advance();
                                        n
                                    }
                                    _ => {
                                        return Err(ParseError::Custom(
                                            "expected index".to_string(),
                                        ))
                                    }
                                };
                                self.expect(Token::Comma)?;
                                let value = self.parse_expr()?;
                                self.expect(Token::RParen)?;
                                Ok(Expr::SimdInsert {
                                    vector: Box::new(vector),
                                    index,
                                    value: Box::new(value),
                                })
                            }
                            _ => {
                                // Parse as generic SIMD intrinsic
                                let op = Self::parse_simd_op(&op_name)?;
                                let args = self.parse_expr_list()?;
                                self.expect(Token::RParen)?;
                                Ok(Expr::SimdIntrinsic { op, args })
                            }
                        }
                    }
                    Some(t) => Err(ParseError::UnexpectedToken {
                        expected: "SIMD operation name".to_string(),
                        found: t,
                        span: self.current_span(),
                    }),
                    None => Err(ParseError::UnexpectedEof),
                }
            }
            Some(t) => Err(ParseError::UnexpectedToken {
                expected: "'[' or '.' after 'simd'".to_string(),
                found: t,
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_simd_op(name: &str) -> ParseResult<SimdOp> {
        match name {
            "add" => Ok(SimdOp::Add),
            "sub" => Ok(SimdOp::Sub),
            "mul" => Ok(SimdOp::Mul),
            "div" => Ok(SimdOp::Div),
            "neg" => Ok(SimdOp::Neg),
            "abs" => Ok(SimdOp::Abs),
            "min" => Ok(SimdOp::Min),
            "max" => Ok(SimdOp::Max),
            "eq" => Ok(SimdOp::Eq),
            "ne" => Ok(SimdOp::Ne),
            "lt" => Ok(SimdOp::Lt),
            "le" => Ok(SimdOp::Le),
            "gt" => Ok(SimdOp::Gt),
            "ge" => Ok(SimdOp::Ge),
            "hadd" => Ok(SimdOp::HAdd),
            "dot" => Ok(SimdOp::Dot),
            "blend" => Ok(SimdOp::Blend),
            "load" => Ok(SimdOp::Load),
            "store" => Ok(SimdOp::Store),
            "load_aligned" => Ok(SimdOp::LoadAligned),
            "store_aligned" => Ok(SimdOp::StoreAligned),
            "cast" => Ok(SimdOp::Cast),
            "widen" => Ok(SimdOp::Widen),
            "narrow" => Ok(SimdOp::Narrow),
            "sqrt" => Ok(SimdOp::Sqrt),
            "rsqrt" => Ok(SimdOp::Rsqrt),
            "rcp" => Ok(SimdOp::Rcp),
            "floor" => Ok(SimdOp::Floor),
            "ceil" => Ok(SimdOp::Ceil),
            "round" => Ok(SimdOp::Round),
            "and" => Ok(SimdOp::And),
            "or" => Ok(SimdOp::Or),
            "xor" => Ok(SimdOp::Xor),
            "not" => Ok(SimdOp::Not),
            "shl" => Ok(SimdOp::Shl),
            "shr" => Ok(SimdOp::Shr),
            _ => Err(ParseError::Custom(format!(
                "unknown SIMD operation: {}",
                name
            ))),
        }
    }

    /// Parse atomic expressions
    ///
    /// Syntax:
    /// ```sigil
    /// atomic.load(ptr, Relaxed)
    /// atomic.store(ptr, value, Release)
    /// atomic.swap(ptr, value, SeqCst)
    /// atomic.compare_exchange(ptr, expected, new, AcqRel, Relaxed)
    /// atomic.fetch_add(ptr, value, Acquire)
    /// atomic.fence(SeqCst)
    /// ```
    fn parse_atomic_expr(&mut self) -> ParseResult<Expr> {
        self.expect(Token::Atomic)?;
        self.expect(Token::Dot)?;

        match self.current_token().cloned() {
            Some(Token::Ident(ref op)) => {
                let op_name = op.clone();
                self.advance();

                if op_name == "fence" {
                    self.expect(Token::LParen)?;
                    let ordering = self.parse_memory_ordering()?;
                    self.expect(Token::RParen)?;
                    return Ok(Expr::AtomicFence { ordering });
                }

                self.expect(Token::LParen)?;
                let ptr = self.parse_expr()?;

                let op = Self::parse_atomic_op(&op_name)?;

                // Parse value for operations that need it
                let value = match op {
                    AtomicOp::Load => None,
                    _ => {
                        self.expect(Token::Comma)?;
                        Some(Box::new(self.parse_expr()?))
                    }
                };

                // Parse expected value for compare_exchange
                let expected = match op {
                    AtomicOp::CompareExchange | AtomicOp::CompareExchangeWeak => {
                        self.expect(Token::Comma)?;
                        Some(Box::new(self.parse_expr()?))
                    }
                    _ => None,
                };

                // Parse memory ordering
                self.expect(Token::Comma)?;
                let ordering = self.parse_memory_ordering()?;

                // Parse failure ordering for compare_exchange
                let failure_ordering = match op {
                    AtomicOp::CompareExchange | AtomicOp::CompareExchangeWeak => {
                        if self.consume_if(&Token::Comma) {
                            Some(self.parse_memory_ordering()?)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                self.expect(Token::RParen)?;

                Ok(Expr::AtomicOp {
                    op,
                    ptr: Box::new(ptr),
                    value,
                    expected,
                    ordering,
                    failure_ordering,
                })
            }
            Some(t) => Err(ParseError::UnexpectedToken {
                expected: "atomic operation name".to_string(),
                found: t,
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_atomic_op(name: &str) -> ParseResult<AtomicOp> {
        match name {
            "load" => Ok(AtomicOp::Load),
            "store" => Ok(AtomicOp::Store),
            "swap" => Ok(AtomicOp::Swap),
            "compare_exchange" => Ok(AtomicOp::CompareExchange),
            "compare_exchange_weak" => Ok(AtomicOp::CompareExchangeWeak),
            "fetch_add" => Ok(AtomicOp::FetchAdd),
            "fetch_sub" => Ok(AtomicOp::FetchSub),
            "fetch_and" => Ok(AtomicOp::FetchAnd),
            "fetch_or" => Ok(AtomicOp::FetchOr),
            "fetch_xor" => Ok(AtomicOp::FetchXor),
            "fetch_min" => Ok(AtomicOp::FetchMin),
            "fetch_max" => Ok(AtomicOp::FetchMax),
            _ => Err(ParseError::Custom(format!(
                "unknown atomic operation: {}",
                name
            ))),
        }
    }

    fn parse_memory_ordering(&mut self) -> ParseResult<MemoryOrdering> {
        match self.current_token() {
            Some(Token::Ident(name)) => {
                let ordering =
                    match name.as_str() {
                        "Relaxed" => MemoryOrdering::Relaxed,
                        "Acquire" => MemoryOrdering::Acquire,
                        "Release" => MemoryOrdering::Release,
                        "AcqRel" => MemoryOrdering::AcqRel,
                        "SeqCst" => MemoryOrdering::SeqCst,
                        _ => return Err(ParseError::Custom(
                            "expected memory ordering (Relaxed, Acquire, Release, AcqRel, SeqCst)"
                                .to_string(),
                        )),
                    };
                self.advance();
                Ok(ordering)
            }
            _ => Err(ParseError::Custom("expected memory ordering".to_string())),
        }
    }

    /// Check if the token after `|` looks like a pipe target (function, closure, morpheme)
    /// rather than a bitwise OR operand (literal, parenthesized expression)
    fn is_pipe_target_ahead(&mut self) -> bool {
        // Peek at the token after the pipe (current token is |)
        if let Some(next) = self.peek_next().cloned() {
            match &next {
                // These indicate a pipe target (function call, closure, morpheme)
                Token::Ident(_) => true,
                Token::SelfLower => true,
                Token::SelfUpper => true,
                // Morpheme operators (τ, φ, σ, ρ, Π, Σ, etc.)
                Token::Tau
                | Token::Phi
                | Token::Sigma
                | Token::Rho
                | Token::Lambda
                | Token::Delta
                | Token::Mu
                | Token::Chi
                | Token::GradeUp
                | Token::GradeDown
                | Token::Rotate
                | Token::Iota
                | Token::ForAll
                | Token::Exists
                | Token::Pi
                | Token::Hourglass => true,
                // Closure syntax |x| or || (lookahead for closure parameter list)
                Token::Pipe => true,
                Token::OrOr => true, // Empty closure ||
                // Move closure
                Token::Move => true,
                // Block expression used as pipe target
                Token::LBrace => true,
                // These indicate bitwise OR (literals, grouping, conditionals)
                Token::IntLit(_)
                | Token::FloatLit(_)
                | Token::HexLit(_)
                | Token::BinaryLit(_)
                | Token::OctalLit(_) => false,
                Token::LParen => false, // Parenthesized expression = bitwise OR
                Token::True | Token::False => false,
                Token::If => false,    // if expression as bitwise OR operand
                Token::Match => false, // match expression as bitwise OR operand
                // Default to pipe for unknown cases
                _ => true,
            }
        } else {
            false // EOF after |, treat as bitwise OR (will error anyway)
        }
    }

    /// Parse postfix operators that can follow a pipe chain (like ?)
    fn parse_postfix_after_pipe(&mut self, mut expr: Expr) -> ParseResult<Expr> {
        loop {
            match self.current_token() {
                Some(Token::Question) => {
                    self.advance();
                    expr = Expr::Try(Box::new(expr));
                }
                Some(Token::Dot) => {
                    self.advance();
                    let field = if let Some(Token::IntLit(idx)) = self.current_token() {
                        let idx = idx.clone();
                        let span = self.current_span();
                        self.advance();
                        Ident {
                            name: idx,
                            evidentiality: None,
                            affect: None,
                            span,
                        }
                    } else {
                        self.parse_ident()?
                    };
                    if self.check(&Token::ColonColon) {
                        self.advance();
                        self.expect(Token::Lt)?;
                        let type_args = self.parse_type_list()?;
                        self.expect_gt()?;
                        self.expect(Token::LParen)?;
                        let args = self.parse_expr_list()?;
                        self.expect(Token::RParen)?;
                        expr = Expr::MethodCall {
                            receiver: Box::new(expr),
                            method: field,
                            type_args: Some(type_args),
                            args,
                        };
                    } else if self.check(&Token::LParen) {
                        self.advance();
                        let args = self.parse_expr_list()?;
                        self.expect(Token::RParen)?;
                        expr = Expr::MethodCall {
                            receiver: Box::new(expr),
                            method: field,
                            type_args: None,
                            args,
                        };
                    } else {
                        expr = Expr::Field {
                            expr: Box::new(expr),
                            field,
                        };
                    }
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    /// Parse a pipe chain: `expr|op1|op2|op3`
    fn parse_pipe_chain(&mut self, initial: Expr) -> ParseResult<Expr> {
        let mut operations = Vec::new();

        while self.consume_if(&Token::Pipe) {
            let op = self.parse_pipe_op()?;
            operations.push(op);
        }

        Ok(Expr::Pipe {
            expr: Box::new(initial),
            operations,
        })
    }

    fn parse_pipe_op(&mut self) -> ParseResult<PipeOp> {
        match self.current_token() {
            Some(Token::Tau) => {
                self.advance();
                self.expect(Token::LBrace)?;
                self.skip_comments();
                // Check for closure pattern: τ{x => expr} or τ{(a, b) => expr}
                let body = if self.looks_like_morpheme_closure() {
                    self.parse_morpheme_closure()?
                } else {
                    self.parse_expr()?
                };
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Transform(Box::new(body)))
            }
            Some(Token::Phi) => {
                self.advance();
                self.expect(Token::LBrace)?;
                self.skip_comments();
                // Check for closure pattern: φ{x => expr} or φ{(a, b) => expr}
                let body = if self.looks_like_morpheme_closure() {
                    self.parse_morpheme_closure()?
                } else {
                    self.parse_expr()?
                };
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Filter(Box::new(body)))
            }
            Some(Token::Sigma) => {
                // Σ can be either sort morpheme OR a function call like Σ(dim: -1)
                if self.peek_next() == Some(&Token::LParen) {
                    // Parse Σ as a function call: Σ(args)
                    let name = Ident {
                        name: "Σ".to_string(),
                        evidentiality: None,
                        affect: None,
                        span: self.current_span(),
                    };
                    self.advance(); // consume Σ
                    self.advance(); // consume (
                    let args = self.parse_expr_list()?;
                    self.expect(Token::RParen)?;
                    Ok(PipeOp::Call(Box::new(Expr::Call {
                        func: Box::new(Expr::Path(TypePath {
                            segments: vec![PathSegment {
                                ident: name,
                                generics: None,
                            }],
                        })),
                        args,
                    })))
                } else {
                    self.advance();
                    let field = if self.consume_if(&Token::Dot) {
                        Some(self.parse_ident()?)
                    } else {
                        None
                    };
                    Ok(PipeOp::Sort(field))
                }
            }
            Some(Token::Rho) => {
                self.advance();
                // Check for reduction variants: ρ+, ρ*, ρ++, ρ&, ρ|, ρ_sum, ρ_prod, ρ_min, ρ_max, ρ_cat, ρ_all, ρ_any
                match self.current_token() {
                    Some(Token::Plus) => {
                        self.advance();
                        Ok(PipeOp::ReduceSum)
                    }
                    Some(Token::Star) => {
                        self.advance();
                        Ok(PipeOp::ReduceProd)
                    }
                    Some(Token::PlusPlus) => {
                        self.advance();
                        Ok(PipeOp::ReduceConcat)
                    }
                    Some(Token::Amp) => {
                        self.advance();
                        Ok(PipeOp::ReduceAll)
                    }
                    Some(Token::Pipe) => {
                        self.advance();
                        Ok(PipeOp::ReduceAny)
                    }
                    Some(Token::Underscore) => {
                        self.advance();
                        // Parse the variant name: _sum, _prod, _min, _max, _cat, _all, _any
                        if let Some(Token::Ident(name)) = self.current_token().cloned() {
                            self.advance();
                            match name.as_str() {
                                "sum" => Ok(PipeOp::ReduceSum),
                                "prod" | "product" => Ok(PipeOp::ReduceProd),
                                "min" => Ok(PipeOp::ReduceMin),
                                "max" => Ok(PipeOp::ReduceMax),
                                "cat" | "concat" => Ok(PipeOp::ReduceConcat),
                                "all" => Ok(PipeOp::ReduceAll),
                                "any" => Ok(PipeOp::ReduceAny),
                                _ => Err(ParseError::Custom(format!(
                                    "unknown reduction variant: ρ_{}",
                                    name
                                ))),
                            }
                        } else {
                            Err(ParseError::Custom(
                                "expected reduction variant name after ρ_".to_string(),
                            ))
                        }
                    }
                    Some(Token::LBrace) => {
                        // General reduce with closure: ρ{(acc, x) => ...}
                        self.advance();
                        self.skip_comments();
                        // Check for closure pattern: ρ{x => expr} or ρ{(a, b) => expr}
                        let body = if self.looks_like_morpheme_closure() {
                            self.parse_morpheme_closure()?
                        } else {
                            self.parse_expr()?
                        };
                        self.expect(Token::RBrace)?;
                        Ok(PipeOp::Reduce(Box::new(body)))
                    }
                    _ => Err(ParseError::Custom(
                        "expected reduction variant (+, *, ++, &, |, _name) or {body} after ρ"
                            .to_string(),
                    )),
                }
            }
            // Product reduction: Π - multiply all elements (shorthand for ρ*)
            Some(Token::Pi) => {
                self.advance();
                Ok(PipeOp::ReduceProd)
            }
            // New access morphemes
            Some(Token::Alpha) => {
                self.advance();
                Ok(PipeOp::First)
            }
            Some(Token::Omega) => {
                self.advance();
                Ok(PipeOp::Last)
            }
            Some(Token::Mu) => {
                // μ can be either the "middle" morpheme OR a function call like μ(axis: -1)
                // If followed by (, it's a function call - parse as normal expression
                if self.peek_next() == Some(&Token::LParen) {
                    // Parse μ as a function call: μ(args) - treat as expression piped call
                    let name = Ident {
                        name: "μ".to_string(),
                        evidentiality: None,
                        affect: None,
                        span: self.current_span(),
                    };
                    self.advance(); // consume μ
                    self.advance(); // consume (
                    let args = self.parse_expr_list()?;
                    self.expect(Token::RParen)?;
                    // Create a call expression and return as PipeOp::Call
                    Ok(PipeOp::Call(Box::new(Expr::Call {
                        func: Box::new(Expr::Path(TypePath {
                            segments: vec![PathSegment {
                                ident: name,
                                generics: None,
                            }],
                        })),
                        args,
                    })))
                } else {
                    self.advance();
                    Ok(PipeOp::Middle)
                }
            }
            Some(Token::Chi) => {
                self.advance();
                Ok(PipeOp::Choice)
            }
            Some(Token::Nu) => {
                self.advance();
                // ν can take an optional index: ν{2}
                if self.check(&Token::LBrace) {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(Token::RBrace)?;
                    Ok(PipeOp::Nth(Box::new(index)))
                } else {
                    // Default to first element if no index given
                    Ok(PipeOp::Nth(Box::new(Expr::Literal(Literal::Int {
                        value: "0".to_string(),
                        base: NumBase::Decimal,
                        suffix: None,
                    }))))
                }
            }
            Some(Token::Xi) => {
                self.advance();
                Ok(PipeOp::Next)
            }
            // Parallel morpheme: ∥τ{f} or parallel τ{f} - wraps another operation
            Some(Token::Parallel) => {
                self.advance();
                // Parse the inner operation to parallelize
                let inner_op = self.parse_pipe_op()?;
                Ok(PipeOp::Parallel(Box::new(inner_op)))
            }
            // GPU compute morpheme: ⊛τ{f} or gpu τ{f} - execute on GPU
            Some(Token::Gpu) => {
                self.advance();
                // Parse the inner operation to run on GPU
                let inner_op = self.parse_pipe_op()?;
                Ok(PipeOp::Gpu(Box::new(inner_op)))
            }
            Some(Token::Await) => {
                self.advance();
                Ok(PipeOp::Await)
            }
            Some(Token::Hourglass) => {
                self.advance();
                Ok(PipeOp::Await)
            }
            Some(Token::MiddleDot) => {
                self.advance();
                let mut prefix = Vec::new();
                prefix.push(self.parse_ident()?);

                while self.consume_if(&Token::MiddleDot) {
                    if self.check(&Token::LBrace) {
                        break;
                    }
                    prefix.push(self.parse_ident()?);
                }

                let body = if self.check(&Token::LBrace) {
                    self.advance();
                    let expr = self.parse_expr()?;
                    self.expect(Token::RBrace)?;
                    Some(Box::new(expr))
                } else {
                    None
                };

                Ok(PipeOp::Named { prefix, body })
            }
            // Match morpheme: |match{ Pattern => expr, ... }
            Some(Token::Match) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let mut arms = Vec::new();
                while !self.check(&Token::RBrace) && !self.is_eof() {
                    // Use parse_or_pattern to support "pat1 | pat2 => expr" arms
                    let pattern = self.parse_or_pattern()?;
                    let guard = if self.consume_if(&Token::If) {
                        Some(self.parse_condition()?)
                    } else {
                        None
                    };
                    self.expect(Token::FatArrow)?;
                    let body = self.parse_expr()?;
                    arms.push(MatchArm {
                        pattern,
                        guard,
                        body,
                    });
                    // Comma is optional after block bodies, just like in regular match
                    self.consume_if(&Token::Comma);
                }
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Match(arms))
            }
            // Trust boundary / unwrap: |‽ or |‽{mapper}
            // Uses interrobang (‽) to signal trust boundary crossing
            Some(Token::Interrobang) => {
                self.advance();
                let mapper = if self.check(&Token::LBrace) {
                    self.advance();
                    let expr = self.parse_expr()?;
                    self.expect(Token::RBrace)?;
                    Some(Box::new(expr))
                } else {
                    None
                };
                Ok(PipeOp::TryMap(mapper))
            }
            // Handle self.field as a pipe call: |self.layer becomes Call(self.layer)
            Some(Token::SelfLower) | Some(Token::SelfUpper) => {
                // Parse self and any field accesses/method calls following it
                let expr = self.parse_postfix_expr()?;
                Ok(PipeOp::Call(Box::new(expr)))
            }
            // Gradient/nabla operator: |∇ - backpropagate gradients
            Some(Token::Nabla) => {
                self.advance();
                // ∇ as a simple gradient pipe - creates gradients from tensor
                Ok(PipeOp::Method {
                    name: Ident {
                        name: "∇".to_string(),
                        evidentiality: None,
                        affect: None,
                        span: self.current_span(),
                    },
                    type_args: None,
                    args: vec![],
                })
            }
            Some(Token::Ident(_)) => {
                let name = self.parse_ident()?;

                // Special handling for evidence promotion operations BEFORE macro check
                // |validate!{predicate} or |validate!(predicate) - validate and promote to Known
                // |validate?{predicate} - validate and promote to Uncertain
                // |validate~{predicate} - validate and keep as Reported
                // |assume!("reason") - assume evidence level
                if name.name == "validate" || name.name == "assume" {
                    // Check for evidentiality marker followed by { or (
                    let (has_marker, target_evidence) = if self.check(&Token::Bang) {
                        let peek = self.peek_next();
                        if matches!(peek, Some(Token::LBrace) | Some(Token::LParen)) {
                            self.advance(); // consume !
                            (true, Evidentiality::Known)
                        } else {
                            (false, Evidentiality::Known)
                        }
                    } else if self.check(&Token::Question) {
                        let peek = self.peek_next();
                        if matches!(peek, Some(Token::LBrace) | Some(Token::LParen)) {
                            self.advance(); // consume ?
                            (true, Evidentiality::Uncertain)
                        } else {
                            (false, Evidentiality::Known)
                        }
                    } else if self.check(&Token::Tilde) {
                        let peek = self.peek_next();
                        if matches!(peek, Some(Token::LBrace) | Some(Token::LParen)) {
                            self.advance(); // consume ~
                            (true, Evidentiality::Reported)
                        } else {
                            (false, Evidentiality::Known)
                        }
                    } else {
                        (false, name.evidentiality.unwrap_or(Evidentiality::Known))
                    };

                    // Check for args - either (args) or {closure}
                    if has_marker || self.check(&Token::LParen) || self.check(&Token::LBrace) {
                        let args = if self.check(&Token::LParen) {
                            self.advance();
                            let args = self.parse_expr_list()?;
                            self.expect(Token::RParen)?;
                            args
                        } else if self.check(&Token::LBrace) {
                            self.advance();
                            self.skip_comments();
                            let body = if self.looks_like_morpheme_closure() {
                                self.parse_morpheme_closure()?
                            } else {
                                self.parse_expr()?
                            };
                            self.expect(Token::RBrace)?;
                            vec![body]
                        } else {
                            vec![]
                        };

                        if name.name == "validate" {
                            if args.is_empty() {
                                return Err(ParseError::Custom(
                                    "validate requires a predicate".to_string(),
                                ));
                            }
                            return Ok(PipeOp::Validate {
                                predicate: Box::new(args.into_iter().next().unwrap()),
                                target_evidence,
                            });
                        } else {
                            // assume
                            let reason = args.into_iter().next().map(Box::new);
                            return Ok(PipeOp::Assume {
                                reason,
                                target_evidence,
                            });
                        }
                    }
                }

                // Check for macro invocation: |macro_name!{ ... } or |macro_name!(...)
                if self.check(&Token::Bang) {
                    let peek = self.peek_next();
                    if matches!(
                        peek,
                        Some(Token::LBrace) | Some(Token::LParen) | Some(Token::LBracket)
                    ) {
                        self.advance(); // consume !
                        let tokens = self.parse_macro_tokens()?;
                        let path = TypePath {
                            segments: vec![PathSegment {
                                ident: name,
                                generics: None,
                            }],
                        };
                        return Ok(PipeOp::Call(Box::new(Expr::Macro { path, tokens })));
                    }
                }

                // Check for path continuation or turbofish syntax:
                // |Tensor::from_slice - path to associated function
                // |collect::<String>() - turbofish generics
                let mut path_segments = vec![PathSegment {
                    ident: name.clone(),
                    generics: None,
                }];
                let type_args = loop {
                    if self.check(&Token::ColonColon) {
                        self.advance(); // consume ::
                        if self.check(&Token::Lt) {
                            // Turbofish: ::<Type>
                            self.advance(); // consume <
                            let types = self.parse_type_list()?;
                            self.expect_gt()?;
                            break Some(types);
                        } else if let Some(Token::Ident(_)) = self.current_token() {
                            // Path continuation: ::segment
                            let segment = self.parse_ident()?;
                            path_segments.push(PathSegment {
                                ident: segment,
                                generics: None,
                            });
                            // Continue to check for more segments or turbofish
                        } else {
                            return Err(ParseError::Custom(
                                "expected identifier or '<' after '::'".to_string(),
                            ));
                        }
                    } else {
                        break None;
                    }
                };

                // If we have a multi-segment path, convert to a path call
                let name = if path_segments.len() > 1 {
                    // Build a Call expression with the full path
                    let path = TypePath {
                        segments: path_segments,
                    };
                    let args = if self.check(&Token::LParen) {
                        self.advance();
                        let args = self.parse_expr_list()?;
                        self.expect(Token::RParen)?;
                        args
                    } else {
                        vec![]
                    };
                    return Ok(PipeOp::Call(Box::new(Expr::Call {
                        func: Box::new(Expr::Path(path)),
                        args,
                    })));
                } else {
                    name
                };
                let args = if self.check(&Token::LParen) {
                    self.advance();
                    let args = self.parse_expr_list()?;
                    self.expect(Token::RParen)?;
                    args
                } else if self.check(&Token::LBrace) && !self.in_condition {
                    // Handle closure-style argument: |method{closure}
                    // But NOT in condition context (for/while/if) where { is the control block
                    self.advance();
                    self.skip_comments();
                    let body = if self.looks_like_morpheme_closure() {
                        self.parse_morpheme_closure()?
                    } else {
                        self.parse_expr()?
                    };
                    self.expect(Token::RBrace)?;
                    vec![body]
                } else {
                    vec![]
                };

                // Special handling for evidence promotion operations
                if name.name == "validate" {
                    let target_evidence = name.evidentiality.unwrap_or(Evidentiality::Known);
                    if args.is_empty() {
                        return Err(ParseError::Custom(
                            "validate requires a predicate: |validate!{predicate}".to_string(),
                        ));
                    }
                    return Ok(PipeOp::Validate {
                        predicate: Box::new(args.into_iter().next().unwrap()),
                        target_evidence,
                    });
                }
                if name.name == "assume" {
                    let target_evidence = name.evidentiality.unwrap_or(Evidentiality::Known);
                    let reason = args.into_iter().next().map(Box::new);
                    return Ok(PipeOp::Assume {
                        reason,
                        target_evidence,
                    });
                }

                Ok(PipeOp::Method {
                    name,
                    type_args,
                    args,
                })
            }

            // ==========================================
            // Protocol Operations - Sigil-native networking
            // ==========================================

            // Send: |send{data} or |⇒{data}
            Some(Token::Send) | Some(Token::ProtoSend) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let data = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Send(Box::new(data)))
            }

            // Recv: |recv or |⇐
            Some(Token::Recv) | Some(Token::ProtoRecv) => {
                self.advance();
                Ok(PipeOp::Recv)
            }

            // Stream: |stream{handler} or |≋{handler}
            Some(Token::Stream) | Some(Token::ProtoStream) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let handler = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Stream(Box::new(handler)))
            }

            // Connect: |connect or |connect{config} or |⊸{config}
            Some(Token::Connect) | Some(Token::ProtoConnect) => {
                self.advance();
                let config = if self.check(&Token::LBrace) {
                    self.advance();
                    let expr = self.parse_expr()?;
                    self.expect(Token::RBrace)?;
                    Some(Box::new(expr))
                } else {
                    None
                };
                Ok(PipeOp::Connect(config))
            }

            // Close: |close or |⊗
            Some(Token::Close) | Some(Token::Tensor) => {
                self.advance();
                Ok(PipeOp::Close)
            }

            // Timeout: |timeout{ms} or |⏱{ms}
            Some(Token::Timeout) | Some(Token::ProtoTimeout) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let ms = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Timeout(Box::new(ms)))
            }

            // Retry: |retry{count} or |retry{count, strategy}
            Some(Token::Retry) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let count = self.parse_expr()?;
                let strategy = if self.consume_if(&Token::Comma) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Retry {
                    count: Box::new(count),
                    strategy,
                })
            }

            // Header: |header{name, value}
            Some(Token::Header) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let name = self.parse_expr()?;
                self.expect(Token::Comma)?;
                let value = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Header {
                    name: Box::new(name),
                    value: Box::new(value),
                })
            }

            // Body: |body{data}
            Some(Token::Body) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let data = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Body(Box::new(data)))
            }

            // ==========================================
            // Mathematical & APL-Inspired Operations
            // ==========================================

            // All/ForAll: |∀{p} for predicate check, |∀ for universal reconstruction
            Some(Token::ForAll) => {
                self.advance();
                // Check if followed by brace - if so, it's All(predicate)
                // Otherwise, it's Universal (holographic reconstruction)
                if self.check(&Token::LBrace) {
                    self.advance(); // consume LBrace
                    let pred = self.parse_expr()?;
                    self.expect(Token::RBrace)?;
                    Ok(PipeOp::All(Box::new(pred)))
                } else {
                    // No brace - universal reconstruction (sum/merge)
                    Ok(PipeOp::Universal)
                }
            }

            // Possibility: |◊ or |◊method - extract approximate/speculative answer
            Some(Token::Lozenge) => {
                self.advance();
                // Check if followed by identifier (method call)
                if let Some(Token::Ident(_)) = self.current_token() {
                    let name = self.parse_ident()?;
                    // Check for arguments
                    let args = if self.check(&Token::LParen) {
                        self.advance();
                        let args = self.parse_expr_list()?;
                        self.expect(Token::RParen)?;
                        args
                    } else {
                        vec![]
                    };
                    Ok(PipeOp::PossibilityMethod { name, args })
                } else {
                    Ok(PipeOp::Possibility)
                }
            }

            // Necessity: |□ or |□method - verify and promote to certain
            Some(Token::BoxSymbol) => {
                self.advance();
                // Check if followed by identifier (method call)
                if let Some(Token::Ident(_)) = self.current_token() {
                    let name = self.parse_ident()?;
                    // Check for arguments
                    let args = if self.check(&Token::LParen) {
                        self.advance();
                        let args = self.parse_expr_list()?;
                        self.expect(Token::RParen)?;
                        args
                    } else {
                        vec![]
                    };
                    Ok(PipeOp::NecessityMethod { name, args })
                } else {
                    Ok(PipeOp::Necessity)
                }
            }

            // Any/Exists: |∃{p} or |any{p}
            Some(Token::Exists) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let pred = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Any(Box::new(pred)))
            }

            // Compose: |∘{f} or |compose{f}
            Some(Token::Compose) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let f = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Compose(Box::new(f)))
            }

            // Zip/Join: |⋈{other} or |zip{other}
            Some(Token::Bowtie) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let other = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Zip(Box::new(other)))
            }

            // Scan/Integral: |∫{f} or |scan{f}
            Some(Token::Integral) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let f = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Scan(Box::new(f)))
            }

            // Diff/Derivative: |∂ or |diff
            Some(Token::Partial) => {
                self.advance();
                Ok(PipeOp::Diff)
            }

            // Gradient: |∇{var} or |grad{var}
            Some(Token::Nabla) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let var = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Gradient(Box::new(var)))
            }

            // Sort Ascending: |⍋ or |sort_asc
            Some(Token::GradeUp) => {
                self.advance();
                Ok(PipeOp::SortAsc)
            }

            // Sort Descending: |⍒ or |sort_desc
            Some(Token::GradeDown) => {
                self.advance();
                Ok(PipeOp::SortDesc)
            }

            // Reverse: |⌽ or |rev
            Some(Token::Rotate) => {
                self.advance();
                Ok(PipeOp::Reverse)
            }

            // Cycle: |↻{n} or |cycle{n}
            Some(Token::CycleArrow) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let n = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Cycle(Box::new(n)))
            }

            // Windows: |⌺{n} or |windows{n}
            Some(Token::QuadDiamond) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let n = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Windows(Box::new(n)))
            }

            // Chunks: |⊞{n} or |chunks{n}
            Some(Token::SquaredPlus) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let n = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Chunks(Box::new(n)))
            }

            // Flatten: |⋳ or |flatten
            Some(Token::ElementSmallVerticalBar) => {
                self.advance();
                Ok(PipeOp::Flatten)
            }

            // Unique: |∪ or |unique
            Some(Token::Union) => {
                self.advance();
                Ok(PipeOp::Unique)
            }

            // Enumerate: |⍳ or |enumerate
            Some(Token::Iota) => {
                self.advance();
                Ok(PipeOp::Enumerate)
            }

            // Reference expression: |&self.field or |&expr
            Some(Token::Amp) => {
                // Parse as expression - the & starts a reference expression
                let expr = self.parse_prefix_expr()?;
                Ok(PipeOp::Call(Box::new(expr)))
            }

            // Direct closure: |{x => body} or |{|args| body}
            // This is a bare pipe-to-closure without morpheme operator
            Some(Token::LBrace) => {
                self.advance();
                self.skip_comments();
                let body = if self.looks_like_morpheme_closure() {
                    self.parse_morpheme_closure()?
                } else {
                    self.parse_expr()?
                };
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Call(Box::new(body)))
            }

            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "pipe operation".to_string(),
                found: token.clone(),
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    /// Check if current position looks like a morpheme closure: ident => or (pattern) =>
    fn looks_like_morpheme_closure(&mut self) -> bool {
        // Simple closure: x => or _ => (may have evidentiality: x~ => or x◊ =>)
        if matches!(
            self.current_token(),
            Some(Token::Ident(_)) | Some(Token::Underscore)
        ) {
            // Check next token - could be => directly or evidentiality marker first
            match self.peek_next() {
                Some(Token::FatArrow) => return true,
                // Evidentiality markers: ident~ =>, ident◊ =>, ident‽ =>
                Some(Token::Tilde) | Some(Token::Lozenge) | Some(Token::Interrobang) => {
                    // Check if => follows the evidentiality marker
                    if matches!(self.peek_n(1), Some(Token::FatArrow)) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        // Reference pattern closure: &x => or &mut x =>
        if matches!(self.current_token(), Some(Token::Amp)) {
            // Look ahead: &x => or &mut x =>
            if matches!(self.peek_next(), Some(Token::Ident(_))) {
                // Check if token after ident is => (possibly with evidentiality in between)
                match self.peek_n(1) {
                    Some(Token::FatArrow) => return true,
                    Some(Token::Tilde) | Some(Token::Lozenge) | Some(Token::Interrobang) => {
                        if matches!(self.peek_n(2), Some(Token::FatArrow)) {
                            return true;
                        }
                    }
                    _ => {}
                }
            } else if matches!(self.peek_next(), Some(Token::Mut)) {
                // &mut x =>
                return true;
            }
        }
        // Tuple pattern closure: (a, b) =>
        // We need to look ahead to find ) followed by =>
        if matches!(self.current_token(), Some(Token::LParen)) {
            // Look ahead to find closing ) and check for =>
            // For simplicity, we'll assume if it starts with ( and contains =>, it's a closure
            return true;
        }
        false
    }

    /// Parse a morpheme closure: x => expr or (a, b) => expr or &x => expr
    /// For morphemes, (a, b) is a SINGLE tuple parameter pattern, not multiple parameters
    fn parse_morpheme_closure(&mut self) -> ParseResult<Expr> {
        let pattern = if self.check(&Token::LParen) {
            // Tuple pattern: (a, b) => expr - treated as single parameter with tuple pattern
            self.advance();
            let mut patterns = Vec::new();
            while !self.check(&Token::RParen) {
                let pat = self.parse_pattern()?;
                patterns.push(pat);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
            self.expect(Token::RParen)?;
            // Create a single tuple pattern
            Pattern::Tuple(patterns)
        } else if self.check(&Token::Amp) {
            // Reference pattern: &x => expr or &mut x => expr
            self.parse_pattern()?
        } else if self.check(&Token::Underscore) {
            // Wildcard pattern: _ => expr
            self.advance();
            Pattern::Wildcard
        } else {
            // Simple pattern: x => expr
            let name = self.parse_ident()?;
            Pattern::Ident {
                mutable: false,
                name,
                evidentiality: None,
            }
        };
        // Accept either => or | as the arrow (for closure-style syntax)
        if !self.consume_if(&Token::FatArrow) {
            self.expect(Token::Pipe)?;
        }
        // Skip comments before body (e.g., // explanation after =>)
        self.skip_comments();
        // Parse body - can be a single expression or a block of statements
        // Multi-line closures have statements + final expression, e.g.:
        //   let x = ...;
        //   y = some_expr;
        //   let z = ...;
        //   result_expr
        // Or compact form: `expr; final_expr`
        let body = {
            // Multi-statement body - parse statements until closing brace
            let mut stmts = Vec::new();
            loop {
                self.skip_comments();
                if self.check(&Token::RBrace) {
                    break;
                }
                if self.check(&Token::Let) {
                    stmts.push(self.parse_let_stmt()?);
                } else if self.check(&Token::Return)
                    || self.check(&Token::Break)
                    || self.check(&Token::Continue)
                {
                    // Control flow - treat as final expression
                    break;
                } else {
                    // Expression (possibly with assignment) - check if statement or final expr
                    let expr = self.parse_expr()?;
                    // Skip comments after expression (before deciding if final)
                    self.skip_comments();
                    if self.consume_if(&Token::Semi) {
                        // Expression statement with semicolon
                        stmts.push(Stmt::Expr(expr));
                    } else if self.check(&Token::RBrace) {
                        // Final expression at end of block
                        if stmts.is_empty() {
                            // Single expression, no block needed
                            return Ok(Expr::Closure {
                                params: vec![ClosureParam { pattern, ty: None }],
                                return_type: None,
                                body: Box::new(expr),
                                is_move: false,
                            });
                        }
                        return Ok(Expr::Closure {
                            params: vec![ClosureParam { pattern, ty: None }],
                            return_type: None,
                            body: Box::new(Expr::Block(Block {
                                stmts,
                                expr: Some(Box::new(expr)),
                            })),
                            is_move: false,
                        });
                    } else {
                        // Continue without semicolon (statement with omitted semi)
                        // This handles: `expr\n let x = ...`
                        stmts.push(Stmt::Expr(expr));
                    }
                }
            }
            // No final expression
            Expr::Block(Block { stmts, expr: None })
        };
        Ok(Expr::Closure {
            params: vec![ClosureParam { pattern, ty: None }],
            return_type: None,
            body: Box::new(body),
            is_move: false,
        })
    }

    fn parse_morpheme_kind(&mut self) -> ParseResult<MorphemeKind> {
        match self.current_token() {
            Some(Token::Tau) => {
                self.advance();
                Ok(MorphemeKind::Transform)
            }
            Some(Token::Phi) => {
                self.advance();
                Ok(MorphemeKind::Filter)
            }
            Some(Token::Sigma) => {
                self.advance();
                Ok(MorphemeKind::Sort)
            }
            Some(Token::Rho) => {
                self.advance();
                Ok(MorphemeKind::Reduce)
            }
            Some(Token::Lambda) => {
                self.advance();
                Ok(MorphemeKind::Lambda)
            }
            Some(Token::Pi) => {
                self.advance();
                Ok(MorphemeKind::Product)
            }
            _ => Err(ParseError::Custom("expected morpheme".to_string())),
        }
    }

    fn parse_block_or_closure(&mut self) -> ParseResult<Expr> {
        self.expect(Token::LBrace)?;
        self.skip_comments();

        // Try to detect closure pattern: `{x => ...}` using lookahead
        // We check if current is Ident and next is FatArrow without consuming tokens
        let is_simple_closure = matches!(self.current_token(), Some(Token::Ident(_)))
            && matches!(self.peek_next(), Some(Token::FatArrow));

        if is_simple_closure {
            let name = self.parse_ident()?;
            self.expect(Token::FatArrow)?;
            self.skip_comments();
            let body = self.parse_expr()?;
            self.skip_comments();
            self.expect(Token::RBrace)?;
            return Ok(Expr::Closure {
                params: vec![ClosureParam {
                    pattern: Pattern::Ident {
                        mutable: false,
                        name,
                        evidentiality: None,
                    },
                    ty: None,
                }],
                return_type: None,
                body: Box::new(body),
                is_move: false,
            });
        }

        // Parse as block
        let mut stmts = Vec::new();
        let mut final_expr = None;

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();
            if self.check(&Token::RBrace) {
                break;
            }

            // Handle statement-level attributes: #[cfg(...)] { ... } or #[attr] let x = ...
            if self.check(&Token::Hash) || self.check(&Token::At) {
                // Parse and collect attributes
                let mut attrs = Vec::new();
                while self.check(&Token::Hash) || self.check(&Token::At) {
                    attrs.push(self.parse_outer_attribute()?);
                    self.skip_comments();
                }

                // After attributes, check what follows
                if self.is_item_start() {
                    // Item with attributes - parse as item
                    let item = self.parse_item()?;
                    stmts.push(Stmt::Item(Box::new(item.node)));
                } else if self.check(&Token::Let) {
                    // Let statement with attributes
                    stmts.push(self.parse_let_stmt()?);
                } else {
                    // Expression with attributes (like #[cfg(...)] { block })
                    let expr = self.parse_expr()?;
                    self.skip_comments();
                    if self.consume_if(&Token::Semi) {
                        stmts.push(Stmt::Semi(expr));
                    } else if self.check(&Token::RBrace) {
                        final_expr = Some(Box::new(expr));
                    } else {
                        stmts.push(Stmt::Expr(expr));
                    }
                }
            } else if self.is_item_start() {
                let item = self.parse_item()?;
                stmts.push(Stmt::Item(Box::new(item.node)));
            } else if self.check(&Token::Let) {
                stmts.push(self.parse_let_stmt()?);
            } else {
                let expr = self.parse_expr()?;
                self.skip_comments();
                if self.consume_if(&Token::Semi) {
                    stmts.push(Stmt::Semi(expr));
                } else if self.check(&Token::RBrace) {
                    final_expr = Some(Box::new(expr));
                } else {
                    stmts.push(Stmt::Expr(expr));
                }
            }
        }

        self.expect(Token::RBrace)?;

        Ok(Expr::Block(Block {
            stmts,
            expr: final_expr,
        }))
    }

    pub(crate) fn parse_block(&mut self) -> ParseResult<Block> {
        match self.parse_block_or_closure()? {
            Expr::Block(block) => Ok(block),
            _ => Err(ParseError::Custom("expected block".to_string())),
        }
    }

    fn parse_let_stmt(&mut self) -> ParseResult<Stmt> {
        self.expect(Token::Let)?;
        let pattern = self.parse_pattern()?;
        let ty = if self.consume_if(&Token::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let init = if self.consume_if(&Token::Eq) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        // Check for let-else pattern: let PATTERN = EXPR else { ... }
        if self.consume_if(&Token::Else) {
            let else_branch = Box::new(Expr::Block(self.parse_block()?));
            // Optionally consume trailing semicolon (valid in Rust: `let ... else { ... };`)
            self.consume_if(&Token::Semi);
            Ok(Stmt::LetElse {
                pattern,
                ty,
                init: init.ok_or_else(|| {
                    ParseError::Custom("let-else requires initializer".to_string())
                })?,
                else_branch,
            })
        } else {
            // Semicolon is optional in Sigil's advanced syntax
            // Consume if present, or allow if next token can start a new statement
            if !self.consume_if(&Token::Semi) {
                if !self.can_start_stmt() && !self.check(&Token::RBrace) {
                    return Err(ParseError::UnexpectedToken {
                        expected: "`;` or new statement".to_string(),
                        found: self.current_token().cloned().unwrap_or(Token::Semi),
                        span: self.current_span(),
                    });
                }
            }
            Ok(Stmt::Let { pattern, ty, init })
        }
    }

    fn parse_if_expr(&mut self) -> ParseResult<Expr> {
        self.expect(Token::If)?;

        // Check for if let pattern = expr form
        let condition = if self.consume_if(&Token::Let) {
            let pattern = self.parse_or_pattern()?;
            self.expect(Token::Eq)?;
            let expr = self.parse_condition()?;
            Expr::Let {
                pattern,
                value: Box::new(expr),
            }
        } else {
            self.parse_condition()?
        };

        let then_branch = self.parse_block()?;
        self.skip_comments(); // Skip comments before else
        let else_branch = if self.consume_if(&Token::Else) {
            if self.check(&Token::If) {
                Some(Box::new(self.parse_if_expr()?))
            } else {
                Some(Box::new(Expr::Block(self.parse_block()?)))
            }
        } else {
            None
        };

        Ok(Expr::If {
            condition: Box::new(condition),
            then_branch,
            else_branch,
        })
    }

    fn parse_match_expr(&mut self) -> ParseResult<Expr> {
        self.expect(Token::Match)?;
        // Use parse_condition to prevent { from being parsed as struct literal
        let expr = self.parse_condition()?;
        self.expect(Token::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_eof() {
            // Skip comments and attributes before match arms: #[cfg(...)]
            loop {
                if matches!(
                    self.current_token(),
                    Some(Token::DocComment(_))
                        | Some(
                            Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_)
                        )
                ) {
                    self.advance();
                } else if self.check(&Token::Hash) {
                    // Skip attribute: #[...]
                    self.skip_attribute()?;
                } else {
                    break;
                }
            }
            if self.check(&Token::RBrace) {
                break;
            }
            let pattern = self.parse_or_pattern()?;
            let guard = if self.consume_if(&Token::If) {
                Some(self.parse_condition()?)
            } else {
                None
            };
            self.expect(Token::FatArrow)?;
            let body = self.parse_expr()?;
            arms.push(MatchArm {
                pattern,
                guard,
                body,
            });
            // In Rust/Sigil, commas are optional after block-bodied match arms
            // So we try to consume a comma, but don't break if absent
            self.consume_if(&Token::Comma);
            // Skip trailing comments after comma or block
            while matches!(
                self.current_token(),
                Some(Token::DocComment(_))
                    | Some(Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_))
            ) {
                self.advance();
            }
        }

        self.expect(Token::RBrace)?;

        Ok(Expr::Match {
            expr: Box::new(expr),
            arms,
        })
    }

    // === Pattern parsing ===

    /// Parse a pattern, handling or-patterns: pat1 | pat2 | pat3
    fn parse_or_pattern(&mut self) -> ParseResult<Pattern> {
        let first = self.parse_pattern()?;

        // Check for | to form or-pattern
        if self.check(&Token::Pipe) {
            let mut patterns = vec![first];
            while self.consume_if(&Token::Pipe) {
                patterns.push(self.parse_pattern()?);
            }
            Ok(Pattern::Or(patterns))
        } else {
            Ok(first)
        }
    }

    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        // Check for prefix evidentiality markers: ?pattern, !pattern, ~pattern, ◊pattern, ‽pattern
        let prefix_ev = match self.current_token() {
            Some(Token::Question) => {
                self.advance();
                Some(Evidentiality::Uncertain)
            }
            Some(Token::Bang) => {
                self.advance();
                Some(Evidentiality::Known)
            }
            Some(Token::Tilde) => {
                self.advance();
                Some(Evidentiality::Reported)
            }
            Some(Token::Lozenge) => {
                self.advance();
                Some(Evidentiality::Predicted)
            }
            Some(Token::Interrobang) => {
                self.advance();
                Some(Evidentiality::Paradox)
            }
            _ => None,
        };

        // If we had a prefix evidentiality, parse the rest of the pattern
        // This handles patterns like `?Some(x)` or `?TypeExpr::Variant { .. }`
        if let Some(ev) = prefix_ev {
            // Parse the inner pattern and wrap it with evidentiality
            let inner_pattern = self.parse_pattern_base()?;
            return match inner_pattern {
                Pattern::Ident {
                    mutable,
                    name,
                    evidentiality: _,
                } => {
                    // Simple identifier pattern with evidentiality
                    Ok(Pattern::Ident {
                        mutable,
                        name,
                        evidentiality: Some(ev),
                    })
                }
                Pattern::Wildcard => {
                    // Convert ?_ to Pattern::Ident with name="_" and evidentiality
                    // This ensures the interpreter can distinguish ?_ from plain _
                    let span = self.current_span();
                    Ok(Pattern::Ident {
                        mutable: false,
                        name: Ident {
                            name: "_".to_string(),
                            evidentiality: None,
                            affect: None,
                            span,
                        },
                        evidentiality: Some(ev),
                    })
                }
                other => {
                    // For complex patterns like enum variants, wrap in an Evidential pattern
                    // Note: This might need adjustment based on AST capabilities
                    // For now, we'll add evidentiality to identifier-based patterns
                    // and pass through complex patterns as-is (the `?` means "if Some")
                    Ok(other)
                }
            };
        }

        self.parse_pattern_base()
    }

    /// Parse a pattern without considering prefix evidentiality markers
    fn parse_pattern_base(&mut self) -> ParseResult<Pattern> {
        match self.current_token().cloned() {
            Some(Token::Underscore) => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            Some(Token::DotDot) => {
                self.advance();
                Ok(Pattern::Rest)
            }
            Some(Token::Mut) => {
                self.advance();
                // Handle `mut self` specially
                let name = if self.check(&Token::SelfLower) {
                    let span = self.current_span();
                    self.advance();
                    Ident {
                        name: "self".to_string(),
                        evidentiality: None,
                        affect: None,
                        span,
                    }
                } else {
                    self.parse_ident()?
                };
                let evidentiality = self.parse_evidentiality_opt();
                Ok(Pattern::Ident {
                    mutable: true,
                    name,
                    evidentiality,
                })
            }
            // Ref pattern: ref ident or ref mut ident (binds by reference)
            Some(Token::Ref) => {
                self.advance();
                let mutable = self.consume_if(&Token::Mut);
                let name = self.parse_ident()?;
                let evidentiality = self.parse_evidentiality_opt();
                Ok(Pattern::RefBinding {
                    mutable,
                    name,
                    evidentiality,
                })
            }
            // Reference pattern: &pattern, &mut pattern, &'a pattern, &'a mut pattern
            Some(Token::Amp) => {
                self.advance();
                // Skip optional lifetime annotation in patterns (e.g., &'a self)
                if matches!(self.current_token(), Some(Token::Lifetime(_))) {
                    self.advance();
                }
                let mutable = self.consume_if(&Token::Mut);
                let inner = self.parse_pattern()?;
                Ok(Pattern::Ref {
                    mutable,
                    pattern: Box::new(inner),
                })
            }
            // Double reference pattern: &&pattern (lexer tokenizes && as AndAnd)
            Some(Token::AndAnd) => {
                self.advance();
                let inner = self.parse_pattern()?;
                // Desugar &&x to &(&x)
                let inner_ref = Pattern::Ref {
                    mutable: false,
                    pattern: Box::new(inner),
                };
                Ok(Pattern::Ref {
                    mutable: false,
                    pattern: Box::new(inner_ref),
                })
            }
            Some(Token::LParen) => {
                self.advance();
                let mut patterns = Vec::new();
                while !self.check(&Token::RParen) {
                    patterns.push(self.parse_pattern()?);
                    if !self.consume_if(&Token::Comma) {
                        break;
                    }
                }
                self.expect(Token::RParen)?;
                // Check for postfix evidentiality on tuple pattern: (a, b)!
                let _ev = self.parse_evidentiality_opt();
                // Note: Pattern::Tuple doesn't have evidentiality field, so we just consume it
                // This allows the let statement to parse correctly
                Ok(Pattern::Tuple(patterns))
            }
            Some(Token::LBracket) => {
                self.advance();
                let mut patterns = Vec::new();
                while !self.check(&Token::RBracket) {
                    patterns.push(self.parse_pattern()?);
                    if !self.consume_if(&Token::Comma) {
                        break;
                    }
                }
                self.expect(Token::RBracket)?;
                Ok(Pattern::Slice(patterns))
            }
            Some(Token::IntLit(_))
            | Some(Token::HexLit(_))
            | Some(Token::OctalLit(_))
            | Some(Token::BinaryLit(_))
            | Some(Token::FloatLit(_))
            | Some(Token::StringLit(_))
            | Some(Token::CharLit(_))
            | Some(Token::True)
            | Some(Token::False)
            | Some(Token::Null) => {
                let lit = self.parse_literal()?;
                // Check for range pattern: lit..end or lit..=end
                if self.check(&Token::DotDot) || self.check(&Token::DotDotEq) {
                    let inclusive = self.consume_if(&Token::DotDotEq);
                    if !inclusive {
                        self.advance(); // consume ..
                    }
                    // Parse end of range if present
                    let end = if matches!(
                        self.current_token(),
                        Some(Token::IntLit(_))
                            | Some(Token::HexLit(_))
                            | Some(Token::OctalLit(_))
                            | Some(Token::BinaryLit(_))
                            | Some(Token::CharLit(_))
                    ) {
                        let end_lit = self.parse_literal()?;
                        Some(Box::new(Pattern::Literal(end_lit)))
                    } else {
                        None
                    };
                    Ok(Pattern::Range {
                        start: Some(Box::new(Pattern::Literal(lit))),
                        end,
                        inclusive,
                    })
                } else {
                    Ok(Pattern::Literal(lit))
                }
            }
            // Handle Self as a pattern (e.g., Self { field1 }, Self::Variant)
            Some(Token::SelfUpper) => {
                let span = self.current_span();
                self.advance();

                // Build a path starting with "Self"
                let mut segments = vec![PathSegment {
                    ident: Ident {
                        name: "Self".to_string(),
                        evidentiality: None,
                        affect: None,
                        span,
                    },
                    generics: None,
                }];

                // Check for path continuation: Self::Variant or Self::Variant::SubVariant
                while self.consume_if(&Token::ColonColon) || self.consume_if(&Token::MiddleDot) {
                    let segment_name = self.parse_ident()?;
                    segments.push(PathSegment {
                        ident: segment_name,
                        generics: None,
                    });
                }

                let path = TypePath { segments };

                // Check for tuple destructuring: Self(x, y) or Self::Variant(x, y)
                if self.check(&Token::LParen) {
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.check(&Token::RParen) {
                        fields.push(self.parse_pattern()?);
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RParen)?;
                    return Ok(Pattern::TupleStruct { path, fields });
                }

                // Check for struct destructuring: Self { field: x } or Self::Variant { field }
                if self.check(&Token::LBrace) {
                    self.advance();
                    let mut fields = Vec::new();
                    let mut rest = false;
                    while !self.check(&Token::RBrace) {
                        while matches!(
                            self.current_token(),
                            Some(Token::DocComment(_))
                                | Some(
                                    Token::LineComment(_)
                                        | Token::TildeComment(_)
                                        | Token::BlockComment(_)
                                )
                        ) {
                            self.advance();
                        }
                        if self.check(&Token::RBrace) {
                            break;
                        }
                        if self.consume_if(&Token::DotDot) {
                            rest = true;
                            if !self.consume_if(&Token::Comma) {
                                break;
                            }
                            continue;
                        }
                        let field_name = self.parse_ident()?;
                        let pattern = if self.consume_if(&Token::Colon) {
                            Some(self.parse_pattern()?)
                        } else {
                            None
                        };
                        fields.push(FieldPattern {
                            name: field_name,
                            pattern,
                        });
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RBrace)?;
                    return Ok(Pattern::Struct { path, fields, rest });
                }

                // Just Self or Self::Variant as a unit pattern
                return Ok(Pattern::Path(path));
            }
            // Handle crate::, self::, super:: path patterns
            Some(Token::Crate) | Some(Token::SelfLower) | Some(Token::Super) => {
                let keyword = self.current_token().cloned();
                let span = self.current_span();
                self.advance();

                // These must be followed by :: for a path pattern
                if !self.consume_if(&Token::ColonColon) && !self.consume_if(&Token::MiddleDot) {
                    // Just `self` as an identifier pattern
                    if matches!(keyword, Some(Token::SelfLower)) {
                        return Ok(Pattern::Ident {
                            mutable: false,
                            name: Ident {
                                name: "self".to_string(),
                                evidentiality: None,
                                affect: None,
                                span,
                            },
                            evidentiality: self.parse_evidentiality_opt(),
                        });
                    }
                    return Err(ParseError::Custom(
                        "expected :: after crate/super in path pattern".to_string(),
                    ));
                }

                // Build the path starting with crate/self/super
                let keyword_name = match keyword {
                    Some(Token::Crate) => "crate",
                    Some(Token::SelfLower) => "self",
                    Some(Token::Super) => "super",
                    _ => unreachable!(),
                };
                let mut segments = vec![PathSegment {
                    ident: Ident {
                        name: keyword_name.to_string(),
                        evidentiality: None,
                        affect: None,
                        span,
                    },
                    generics: None,
                }];

                // Parse remaining path segments
                loop {
                    let segment_name = self.parse_ident()?;
                    segments.push(PathSegment {
                        ident: segment_name,
                        generics: None,
                    });

                    if !self.consume_if(&Token::ColonColon) && !self.consume_if(&Token::MiddleDot) {
                        break;
                    }
                }

                let path = TypePath { segments };

                // Check for tuple destructuring: crate::module::Variant(x)
                if self.check(&Token::LParen) {
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.check(&Token::RParen) {
                        fields.push(self.parse_pattern()?);
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RParen)?;
                    return Ok(Pattern::TupleStruct { path, fields });
                }

                // Check for struct destructuring: crate::module::Variant { field: x }
                if self.check(&Token::LBrace) {
                    self.advance();
                    let mut fields = Vec::new();
                    let mut rest = false;
                    while !self.check(&Token::RBrace) {
                        while matches!(
                            self.current_token(),
                            Some(Token::DocComment(_))
                                | Some(
                                    Token::LineComment(_)
                                        | Token::TildeComment(_)
                                        | Token::BlockComment(_)
                                )
                        ) {
                            self.advance();
                        }
                        if self.check(&Token::RBrace) {
                            break;
                        }
                        if self.consume_if(&Token::DotDot) {
                            rest = true;
                            if !self.consume_if(&Token::Comma) {
                                break;
                            }
                            continue;
                        }
                        let field_name = self.parse_ident()?;
                        let pattern = if self.consume_if(&Token::Colon) {
                            Some(self.parse_pattern()?)
                        } else {
                            None
                        };
                        fields.push(FieldPattern {
                            name: field_name,
                            pattern,
                        });
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RBrace)?;
                    return Ok(Pattern::Struct { path, fields, rest });
                }

                // Just a path pattern (unit variant)
                return Ok(Pattern::Path(path));
            }
            Some(Token::Ident(_)) => {
                let name = self.parse_ident()?;

                // Check for path continuation :: to form qualified path (e.g., Token::Fn)
                if self.consume_if(&Token::ColonColon) || self.consume_if(&Token::MiddleDot) {
                    // Build a path pattern
                    let mut segments = vec![PathSegment {
                        ident: name,
                        generics: None,
                    }];

                    // Parse remaining path segments
                    loop {
                        let segment_name = self.parse_ident()?;
                        segments.push(PathSegment {
                            ident: segment_name,
                            generics: None,
                        });

                        if !self.consume_if(&Token::ColonColon)
                            && !self.consume_if(&Token::MiddleDot)
                        {
                            break;
                        }
                    }

                    let path = TypePath { segments };

                    // Check for tuple destructuring: Token::IntLit(x)
                    if self.check(&Token::LParen) {
                        self.advance();
                        let mut fields = Vec::new();
                        while !self.check(&Token::RParen) {
                            fields.push(self.parse_pattern()?);
                            if !self.consume_if(&Token::Comma) {
                                break;
                            }
                        }
                        self.expect(Token::RParen)?;
                        return Ok(Pattern::TupleStruct { path, fields });
                    }

                    // Check for struct destructuring: Token::SomeVariant { field: x }
                    if self.check(&Token::LBrace) {
                        self.advance();
                        let mut fields = Vec::new();
                        let mut rest = false;
                        while !self.check(&Token::RBrace) {
                            // Skip comments
                            while matches!(
                                self.current_token(),
                                Some(Token::DocComment(_))
                                    | Some(
                                        Token::LineComment(_)
                                            | Token::TildeComment(_)
                                            | Token::BlockComment(_)
                                    )
                            ) {
                                self.advance();
                            }
                            if self.check(&Token::RBrace) {
                                break;
                            }

                            // Check for rest pattern: ..
                            if self.consume_if(&Token::DotDot) {
                                rest = true;
                                if !self.consume_if(&Token::Comma) {
                                    break;
                                }
                                continue;
                            }

                            let field_name = self.parse_ident()?;
                            let pattern = if self.consume_if(&Token::Colon) {
                                Some(self.parse_pattern()?)
                            } else {
                                // Shorthand: field punning
                                None
                            };
                            fields.push(FieldPattern {
                                name: field_name,
                                pattern,
                            });

                            if !self.consume_if(&Token::Comma) {
                                break;
                            }
                        }
                        self.expect(Token::RBrace)?;
                        return Ok(Pattern::Struct { path, fields, rest });
                    }

                    // Just a path pattern (unit variant)
                    return Ok(Pattern::Path(path));
                }

                // Check for struct pattern with simple identifier: Foo { ... }
                if self.check(&Token::LBrace) {
                    // Single-segment path for struct pattern
                    let path = TypePath {
                        segments: vec![PathSegment {
                            ident: name,
                            generics: None,
                        }],
                    };
                    self.advance();
                    let mut fields = Vec::new();
                    let mut rest = false;
                    while !self.check(&Token::RBrace) {
                        while matches!(
                            self.current_token(),
                            Some(Token::DocComment(_))
                                | Some(
                                    Token::LineComment(_)
                                        | Token::TildeComment(_)
                                        | Token::BlockComment(_)
                                )
                        ) {
                            self.advance();
                        }
                        if self.check(&Token::RBrace) {
                            break;
                        }
                        if self.consume_if(&Token::DotDot) {
                            rest = true;
                            break;
                        }
                        let field_name = self.parse_ident()?;
                        let pattern = if self.consume_if(&Token::Colon) {
                            Some(self.parse_pattern()?)
                        } else {
                            None
                        };
                        fields.push(FieldPattern {
                            name: field_name,
                            pattern,
                        });
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RBrace)?;
                    return Ok(Pattern::Struct { path, fields, rest });
                }

                // Check for tuple struct pattern with simple identifier: Foo(x, y)
                if self.check(&Token::LParen) {
                    let path = TypePath {
                        segments: vec![PathSegment {
                            ident: name,
                            generics: None,
                        }],
                    };
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.check(&Token::RParen) {
                        fields.push(self.parse_pattern()?);
                        if !self.consume_if(&Token::Comma) {
                            break;
                        }
                    }
                    self.expect(Token::RParen)?;
                    return Ok(Pattern::TupleStruct { path, fields });
                }

                // Simple identifier pattern
                let evidentiality = self.parse_evidentiality_opt();
                Ok(Pattern::Ident {
                    mutable: false,
                    name,
                    evidentiality,
                })
            }
            Some(Token::SelfLower) => {
                // self keyword as pattern in method parameters
                let span = self.current_span();
                self.advance();
                Ok(Pattern::Ident {
                    mutable: false,
                    name: Ident {
                        name: "self".to_string(),
                        evidentiality: None,
                        affect: None,
                        span,
                    },
                    evidentiality: None,
                })
            }
            // Handle contextual keywords as identifiers in patterns
            Some(ref token) if Self::keyword_as_ident(token).is_some() => {
                let name = self.parse_ident()?;
                let evidentiality = self.parse_evidentiality_opt();
                Ok(Pattern::Ident {
                    mutable: false,
                    name,
                    evidentiality,
                })
            }
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "pattern".to_string(),
                found: token,
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_literal(&mut self) -> ParseResult<Literal> {
        match self.current_token().cloned() {
            Some(Token::IntLit(s)) => {
                self.advance();
                Ok(Literal::Int {
                    value: s,
                    base: NumBase::Decimal,
                    suffix: None,
                })
            }
            Some(Token::HexLit(s)) => {
                self.advance();
                Ok(Literal::Int {
                    value: s,
                    base: NumBase::Hex,
                    suffix: None,
                })
            }
            Some(Token::OctalLit(s)) => {
                self.advance();
                Ok(Literal::Int {
                    value: s,
                    base: NumBase::Octal,
                    suffix: None,
                })
            }
            Some(Token::BinaryLit(s)) => {
                self.advance();
                Ok(Literal::Int {
                    value: s,
                    base: NumBase::Binary,
                    suffix: None,
                })
            }
            Some(Token::FloatLit(s)) => {
                self.advance();
                Ok(Literal::Float {
                    value: s,
                    suffix: None,
                })
            }
            Some(Token::StringLit(s)) => {
                self.advance();
                Ok(Literal::String(s))
            }
            Some(Token::CharLit(c)) => {
                self.advance();
                Ok(Literal::Char(c))
            }
            Some(Token::True) => {
                self.advance();
                Ok(Literal::Bool(true))
            }
            Some(Token::False) => {
                self.advance();
                Ok(Literal::Bool(false))
            }
            Some(Token::Null) => {
                self.advance();
                Ok(Literal::Null)
            }
            _ => Err(ParseError::Custom("expected literal".to_string())),
        }
    }

    // === Helpers ===

    /// Convert an expression to an IncorporationSegment for polysynthetic chains
    /// E.g., `path` in `path·file·read` becomes IncorporationSegment { name: "path", args: None }
    fn expr_to_incorporation_segment(&self, expr: Expr) -> ParseResult<IncorporationSegment> {
        match expr {
            Expr::Path(path) if path.segments.len() == 1 => Ok(IncorporationSegment {
                name: path.segments[0].ident.clone(),
                args: None,
            }),
            Expr::Call { func, args } => {
                match *func {
                    Expr::Path(path) => {
                        // For paths like `use_context[T]()` or `serde_json::from_value(s)`,
                        // use the last segment as the incorporation name
                        if let Some(last_seg) = path.segments.last() {
                            return Ok(IncorporationSegment {
                                name: last_seg.ident.clone(),
                                args: Some(args),
                            });
                        }
                        Err(ParseError::Custom(
                            "incorporation chain: empty path".to_string(),
                        ))
                    }
                    // Handle method calls like obj.method()·chain
                    Expr::Field { expr, field } => Ok(IncorporationSegment {
                        name: field.clone(),
                        args: Some(std::iter::once(*expr).chain(args).collect()),
                    }),
                    _ => Err(ParseError::Custom(
                        "incorporation chain must start with identifier or call".to_string(),
                    )),
                }
            }
            // Field access can start an incorporation chain: ctx.navigate·clone()
            // We keep the field name and store the object expression as an argument
            Expr::Field { expr, field } => {
                // Use the field name directly, and store the base object as an argument
                // so the interpreter can reconstruct the field access
                Ok(IncorporationSegment {
                    name: field.clone(),
                    args: Some(vec![*expr]), // The object becomes the implicit "self" argument
                })
            }
            // For literals like "string"·to_string(), use a synthetic "__lit__" segment
            // The literal itself becomes the argument
            Expr::Literal(_) => Ok(IncorporationSegment {
                name: Ident {
                    name: "__lit__".to_string(),
                    evidentiality: None,
                    affect: None,
                    span: crate::span::Span::default(),
                },
                args: Some(vec![expr]),
            }),
            // For unary expressions like *self.field·method(), use a synthetic segment
            // The unary expression becomes the argument
            Expr::Unary { .. } => Ok(IncorporationSegment {
                name: Ident {
                    name: "__unary__".to_string(),
                    evidentiality: None,
                    affect: None,
                    span: crate::span::Span::default(),
                },
                args: Some(vec![expr]),
            }),
            // For index expressions like arr[i]·method()
            Expr::Index { expr: base, index } => Ok(IncorporationSegment {
                name: Ident {
                    name: "__index__".to_string(),
                    evidentiality: None,
                    affect: None,
                    span: crate::span::Span::default(),
                },
                args: Some(vec![*base, *index]),
            }),
            // For any other expression types (if, match, block, closure, etc.)
            // Use a synthetic segment to hold the expression as the receiver
            other => Ok(IncorporationSegment {
                name: Ident {
                    name: "__expr__".to_string(),
                    evidentiality: None,
                    affect: None,
                    span: crate::span::Span::default(),
                },
                args: Some(vec![other]),
            }),
        }
    }

    /// Convert a keyword token to its string name for contextual use as identifier
    fn keyword_as_ident(token: &Token) -> Option<&'static str> {
        match token {
            // Common keywords that may be used as field/variable names
            Token::Packed => Some("packed"),
            Token::As => Some("as"),
            Token::Type => Some("type"),
            Token::Crate => Some("crate"),
            Token::Super => Some("super"),
            Token::Mod => Some("mod"),
            Token::Use => Some("use"),
            Token::Pub => Some("pub"),
            Token::Const => Some("const"),
            Token::Static => Some("static"),
            Token::Extern => Some("extern"),
            Token::Unsafe => Some("unsafe"),
            Token::Async => Some("async"),
            Token::Await => Some("await"),
            Token::Move => Some("move"),
            Token::Dyn => Some("dyn"),
            Token::Atomic => Some("atomic"),
            Token::Volatile => Some("volatile"),
            Token::Naked => Some("naked"),
            Token::Connect => Some("connect"),
            Token::Close => Some("close"),
            Token::Simd => Some("simd"),
            Token::Derive => Some("derive"),
            Token::On => Some("on"),
            Token::Send => Some("send"),
            Token::Recv => Some("recv"),
            Token::Stream => Some("stream"),
            Token::Timeout => Some("timeout"),
            Token::Retry => Some("retry"),
            Token::Header => Some("header"),
            Token::Body => Some("body"),
            Token::Http => Some("http"),
            Token::Https => Some("https"),
            Token::Ws => Some("ws"),
            Token::Wss => Some("wss"),
            Token::Grpc => Some("grpc"),
            Token::Kafka => Some("kafka"),
            Token::Amqp => Some("amqp"),
            Token::GraphQL => Some("graphql"),
            Token::Actor => Some("actor"),
            Token::Saga => Some("saga"),
            Token::Scope => Some("scope"),
            Token::Rune => Some("rune"),
            // Plurality keywords - usable as identifiers in most contexts
            Token::Split => Some("split"),
            Token::Trigger => Some("trigger"),
            Token::Location => Some("location"),
            Token::States => Some("states"),
            Token::To => Some("to"),
            Token::From => Some("from"),
            Token::Headspace => Some("headspace"),
            Token::CoCon => Some("cocon"),
            Token::Reality => Some("reality"),
            Token::Layer => Some("layer"),
            Token::Anima => Some("anima"),
            Token::Struct => Some("sigil"), // Allow 'sigil' as identifier (maps to Struct token)
            // Greek morpheme tokens - allow as identifiers in type/variable contexts
            Token::Parallel => Some("Parallel"),
            Token::Nu => Some("Nu"),
            Token::Lambda => Some("Lambda"),
            Token::Delta => Some("Delta"),
            Token::Tau => Some("Tau"),
            Token::Phi => Some("Phi"),
            Token::Sigma => Some("Sigma"),
            Token::Rho => Some("Rho"),
            Token::Pi => Some("Pi"),
            Token::Epsilon => Some("Epsilon"),
            Token::Omega => Some("Omega"),
            Token::Alpha => Some("Alpha"),
            Token::Zeta => Some("Zeta"),
            Token::Mu => Some("Mu"),
            Token::Chi => Some("Chi"),
            Token::Xi => Some("Xi"),
            Token::Psi => Some("Psi"),
            Token::Theta => Some("Theta"),
            Token::Kappa => Some("Kappa"),
            Token::Nabla => Some("∇"),
            Token::Gpu => Some("Gpu"),
            // Legion/communication operators - can be used as identifiers
            Token::Broadcast => Some("broadcast"),
            Token::Gather => Some("gather"),
            Token::Distribute => Some("distribute"),
            Token::Interfere => Some("interfere"),
            Token::Consensus => Some("consensus"),
            // Other contextual keywords
            Token::Ref => Some("ref"),
            Token::Null => Some("null"),
            _ => None,
        }
    }

    pub(crate) fn parse_ident(&mut self) -> ParseResult<Ident> {
        match self.current.take() {
            Some((Token::Ident(name), span)) => {
                self.current = self.lexer.next_token();
                // Parse optional UNAMBIGUOUS evidentiality markers after identifier: field◊, value~
                // NOTE: Don't consume ! or ? here as they have other meanings (macro!/try?)
                let evidentiality = self.parse_unambiguous_evidentiality_opt();
                // Parse optional affective markers after identifier
                let affect = self.parse_affect_opt();
                Ok(Ident {
                    name,
                    evidentiality,
                    affect,
                    span,
                })
            }
            Some((ref token, span)) if Self::keyword_as_ident(token).is_some() => {
                let mut name = Self::keyword_as_ident(token).unwrap().to_string();
                self.current = self.lexer.next_token();
                // Check if next token is an identifier starting with underscore
                // This handles Greek letter + underscore patterns like λ_Pipeline
                if let Some((Token::Ident(suffix), suffix_span)) = &self.current {
                    if suffix.starts_with('_') {
                        name.push_str(suffix);
                        let merged_span = span.merge(*suffix_span);
                        self.current = self.lexer.next_token();
                        let evidentiality = self.parse_unambiguous_evidentiality_opt();
                        let affect = self.parse_affect_opt();
                        return Ok(Ident {
                            name,
                            evidentiality,
                            affect,
                            span: merged_span,
                        });
                    }
                }
                let evidentiality = self.parse_unambiguous_evidentiality_opt();
                let affect = self.parse_affect_opt();
                Ok(Ident {
                    name,
                    evidentiality,
                    affect,
                    span,
                })
            }
            Some((token, span)) => {
                self.current = Some((token.clone(), span));
                Err(ParseError::UnexpectedToken {
                    expected: "identifier".to_string(),
                    found: token,
                    span,
                })
            }
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_evidentiality_opt(&mut self) -> Option<Evidentiality> {
        // Parse one or more evidentiality markers (e.g., !! or ?!)
        // Multiple markers are combined, with the last one taking precedence
        let mut ev = None;
        loop {
            match self.current_token() {
                Some(Token::Bang) => {
                    self.advance();
                    ev = Some(Evidentiality::Known);
                }
                Some(Token::Question) => {
                    self.advance();
                    ev = Some(Evidentiality::Uncertain);
                }
                Some(Token::Tilde) => {
                    self.advance();
                    ev = Some(Evidentiality::Reported);
                }
                Some(Token::Lozenge) => {
                    self.advance();
                    ev = Some(Evidentiality::Predicted);
                }
                Some(Token::Interrobang) => {
                    self.advance();
                    ev = Some(Evidentiality::Paradox);
                }
                _ => break,
            }
        }
        ev
    }

    /// Parse UNAMBIGUOUS evidentiality markers only: ~, ◊, ‽
    /// Does NOT consume ! or ? as they have other meanings (macro!/try?)
    fn parse_unambiguous_evidentiality_opt(&mut self) -> Option<Evidentiality> {
        let mut ev = None;
        loop {
            match self.current_token() {
                Some(Token::Tilde) => {
                    self.advance();
                    ev = Some(Evidentiality::Reported);
                }
                Some(Token::Lozenge) => {
                    self.advance();
                    ev = Some(Evidentiality::Predicted);
                }
                Some(Token::Interrobang) => {
                    self.advance();
                    ev = Some(Evidentiality::Paradox);
                }
                _ => break,
            }
        }
        ev
    }

    /// Parse optional affective markers: sentiment, sarcasm, intensity, formality, emotion, confidence
    /// Symbols: ⊕ ⊖ ⊜ (sentiment), ⸮ (sarcasm), ↑ ↓ ⇈ (intensity), ♔ ♟ (formality),
    ///          ☺ ☹ ⚡ ❄ ✦ ♡ (emotions), ◉ ◎ ○ (confidence)
    fn parse_affect_opt(&mut self) -> Option<Affect> {
        let mut sentiment = None;
        let mut sarcasm = false;
        let mut intensity = None;
        let mut formality = None;
        let mut emotion = None;
        let mut confidence = None;
        let mut found_any = false;

        // Parse all consecutive affective markers
        loop {
            match self.current_token() {
                // Sentiment markers
                Some(Token::DirectSum) => {
                    self.advance();
                    sentiment = Some(Sentiment::Positive);
                    found_any = true;
                }
                Some(Token::AffectNegative) => {
                    self.advance();
                    sentiment = Some(Sentiment::Negative);
                    found_any = true;
                }
                Some(Token::AffectNeutral) => {
                    self.advance();
                    sentiment = Some(Sentiment::Neutral);
                    found_any = true;
                }
                // Sarcasm/Irony
                Some(Token::IronyMark) => {
                    self.advance();
                    sarcasm = true;
                    found_any = true;
                }
                // Intensity
                Some(Token::IntensityUp) => {
                    self.advance();
                    intensity = Some(Intensity::Up);
                    found_any = true;
                }
                Some(Token::IntensityDown) => {
                    self.advance();
                    intensity = Some(Intensity::Down);
                    found_any = true;
                }
                Some(Token::IntensityMax) => {
                    self.advance();
                    intensity = Some(Intensity::Max);
                    found_any = true;
                }
                // Formality
                Some(Token::FormalRegister) => {
                    self.advance();
                    formality = Some(Formality::Formal);
                    found_any = true;
                }
                Some(Token::InformalRegister) => {
                    self.advance();
                    formality = Some(Formality::Informal);
                    found_any = true;
                }
                // Emotions (Plutchik's wheel)
                Some(Token::EmotionJoy) => {
                    self.advance();
                    emotion = Some(Emotion::Joy);
                    found_any = true;
                }
                Some(Token::EmotionSadness) => {
                    self.advance();
                    emotion = Some(Emotion::Sadness);
                    found_any = true;
                }
                Some(Token::EmotionAnger) => {
                    self.advance();
                    emotion = Some(Emotion::Anger);
                    found_any = true;
                }
                Some(Token::EmotionFear) => {
                    self.advance();
                    emotion = Some(Emotion::Fear);
                    found_any = true;
                }
                Some(Token::EmotionSurprise) => {
                    self.advance();
                    emotion = Some(Emotion::Surprise);
                    found_any = true;
                }
                Some(Token::EmotionLove) => {
                    self.advance();
                    emotion = Some(Emotion::Love);
                    found_any = true;
                }
                // Confidence
                Some(Token::ConfidenceHigh) => {
                    self.advance();
                    confidence = Some(Confidence::High);
                    found_any = true;
                }
                Some(Token::ConfidenceMedium) => {
                    self.advance();
                    confidence = Some(Confidence::Medium);
                    found_any = true;
                }
                Some(Token::ConfidenceLow) => {
                    self.advance();
                    confidence = Some(Confidence::Low);
                    found_any = true;
                }
                _ => break,
            }
        }

        if found_any {
            Some(Affect {
                sentiment,
                sarcasm,
                intensity,
                formality,
                emotion,
                confidence,
            })
        } else {
            None
        }
    }

    pub(crate) fn parse_generics_opt(&mut self) -> ParseResult<Option<Generics>> {
        // Support both <T> and [T] syntax for generics
        let use_brackets = if self.consume_if(&Token::Lt) {
            false
        } else if self.consume_if(&Token::LBracket) {
            true
        } else {
            return Ok(None);
        };

        let mut params = Vec::new();
        // Use check_gt() to handle pending `>` from split `>>` (only for angle brackets)
        while !self.is_eof() {
            // Skip line comments between generic parameters
            self.skip_comments();

            if use_brackets {
                if self.check(&Token::RBracket) {
                    break;
                }
            } else if self.check_gt() {
                break;
            }

            // Check for lifetime parameter: 'a, 'static, etc.
            if let Some(Token::Lifetime(lt)) = self.current_token().cloned() {
                self.advance();
                params.push(GenericParam::Lifetime(lt));
                if !self.consume_if(&Token::Comma) {
                    break;
                }
                continue;
            }

            // Check for const generic parameter (const N: usize = 10 or just const N)
            if self.consume_if(&Token::Const) {
                let name = self.parse_ident()?;
                // Type annotation is optional: const N: usize or just const N
                let ty = if self.consume_if(&Token::Colon) {
                    self.parse_type()?
                } else {
                    TypeExpr::Infer
                };
                // Parse optional default value: const N: usize = 10
                let default = if self.consume_if(&Token::Eq) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                params.push(GenericParam::Const { name, ty, default });
                if !self.consume_if(&Token::Comma) {
                    break;
                }
                continue;
            }

            // Type parameter
            let name = self.parse_ident()?;
            let evidentiality = self.parse_evidentiality_opt();
            let bounds = if self.consume_if(&Token::Colon) {
                self.parse_type_bounds()?
            } else {
                vec![]
            };
            // Parse optional default type: T = DefaultType
            let default = if self.consume_if(&Token::Eq) {
                Some(self.parse_type()?)
            } else {
                None
            };
            params.push(GenericParam::Type {
                name,
                bounds,
                evidentiality,
                default,
            });

            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        // Close the generics
        if use_brackets {
            self.expect(Token::RBracket)?;
        } else {
            // Use expect_gt() to handle nested generics with `>>`
            self.expect_gt()?;
        }

        Ok(Some(Generics { params }))
    }

    pub(crate) fn parse_where_clause_opt(&mut self) -> ParseResult<Option<WhereClause>> {
        if !self.consume_if(&Token::Where) {
            return Ok(None);
        }

        let mut predicates = Vec::new();
        loop {
            self.skip_comments(); // Skip comments between predicates

            // Check for expression predicates: EXPR == EXPR (e.g., QH % KVH == 0)
            // These start with an identifier followed by an operator other than :
            // Try to detect expression predicate by peeking ahead
            // Expression predicates have operators like %, ==, !=, <, >, etc.
            let is_expr_predicate = if let Some(Token::Ident(_)) = self.current_token() {
                // Peek ahead to see if we have an expression operator instead of :
                let next = self.peek_next();
                matches!(
                    next,
                    Some(Token::Percent)
                        | Some(Token::EqEq)
                        | Some(Token::NotEq)
                        | Some(Token::Plus)
                        | Some(Token::Minus)
                        | Some(Token::Star)
                        | Some(Token::Slash)
                )
            } else {
                false
            };

            if is_expr_predicate {
                // Parse as expression predicate: EXPR == EXPR or EXPR != EXPR
                let _expr = self.parse_expr()?;
                // For now, just skip the constraint - we'll store it properly later
                self.skip_comments();
                if !self.consume_if(&Token::Comma) {
                    break;
                }
                self.skip_comments();
                if self.check(&Token::LBrace) {
                    break;
                }
                continue;
            }

            let ty = self.parse_type()?;

            // Check if this is a type bound (T: Trait) or if we should skip
            if self.check(&Token::Colon) {
                self.advance(); // consume :
                let bounds = self.parse_type_bounds()?;
                self.skip_comments(); // Skip comments after bounds
                predicates.push(WherePredicate { ty, bounds });
            } else {
                // Not a type bound, might be an expression we couldn't detect earlier
                // Just skip this predicate
            }

            if !self.consume_if(&Token::Comma) {
                break;
            }
            // Stop if we hit a brace
            self.skip_comments();
            if self.check(&Token::LBrace) {
                break;
            }
        }

        Ok(Some(WhereClause { predicates }))
    }

    pub(crate) fn parse_params(&mut self) -> ParseResult<Vec<Param>> {
        let mut params = Vec::new();
        while !self.check(&Token::RParen) && !self.is_eof() {
            // Skip comments and parameter attributes: @[...] or #[...]
            self.skip_comments();
            while self.check(&Token::At) || self.check(&Token::Hash) {
                self.skip_attribute()?;
                self.skip_comments();
            }
            if self.check(&Token::RParen) {
                break;
            }
            let pattern = self.parse_pattern()?;
            // Type annotation is optional - use Infer if not provided
            let ty = if self.consume_if(&Token::Colon) {
                self.parse_type()?
            } else {
                TypeExpr::Infer
            };
            params.push(Param { pattern, ty });
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        Ok(params)
    }

    /// Skip an attribute (@[...] or #[...]) without parsing its contents
    fn skip_attribute(&mut self) -> ParseResult<()> {
        // Consume @ or #
        if self.check(&Token::At) || self.check(&Token::Hash) {
            self.advance();
        }
        // Consume [...] if present
        if self.consume_if(&Token::LBracket) {
            let mut depth = 1;
            while depth > 0 && !self.is_eof() {
                match self.current_token() {
                    Some(Token::LBracket) => depth += 1,
                    Some(Token::RBracket) => depth -= 1,
                    _ => {}
                }
                self.advance();
            }
        }
        Ok(())
    }

    fn parse_field_defs(&mut self) -> ParseResult<Vec<FieldDef>> {
        let mut fields = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_eof() {
            // Skip doc comments, line comments, and attributes before fields
            while matches!(
                self.current_token(),
                Some(Token::DocComment(_))
                    | Some(Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_))
            ) || self.check(&Token::Hash)
                || self.check(&Token::At)
            {
                if self.check(&Token::Hash) || self.check(&Token::At) {
                    self.skip_attribute()?;
                } else {
                    self.advance();
                }
            }
            if self.check(&Token::RBrace) {
                break;
            }
            let visibility = self.parse_visibility()?;
            let name = self.parse_ident()?;
            // Optional evidentiality marker after field name: `field~: Type`, `field◊: Type`
            let _evidentiality = self.parse_evidentiality_opt();
            self.expect(Token::Colon)?;
            let ty = self.parse_type()?;
            // Parse optional default value: `field: Type = default_expr`
            let default = if self.consume_if(&Token::Eq) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            fields.push(FieldDef {
                visibility,
                name,
                ty,
                default,
            });
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        Ok(fields)
    }

    fn parse_expr_list(&mut self) -> ParseResult<Vec<Expr>> {
        let mut exprs = Vec::new();
        // Skip leading comments
        self.skip_comments();
        while !self.check(&Token::RParen) && !self.check(&Token::RBracket) && !self.is_eof() {
            // Check for named argument syntax: name: expr
            // This handles calls like `stack(axis: 0)` or `func(x: 1, y: 2)`
            // We detect `ident :` but not `ident ::` (path separator)
            let expr = if let Some(Token::Ident(name)) = self.current_token().cloned() {
                // Look ahead: is next token `:` (single colon) not `::` (double colon)?
                let is_named_arg = self.peek_next() == Some(&Token::Colon);
                if is_named_arg {
                    let span = self.current_span();
                    self.advance(); // consume name
                    self.advance(); // consume :
                    let value = self.parse_expr()?;
                    // Represent as a named argument via NamedArg expression
                    Expr::NamedArg {
                        name: Ident {
                            name,
                            evidentiality: None,
                            affect: None,
                            span,
                        },
                        value: Box::new(value),
                    }
                } else {
                    self.parse_expr()?
                }
            } else {
                self.parse_expr()?
            };
            exprs.push(expr);
            if !self.consume_if(&Token::Comma) {
                break;
            }
            // Skip comments after comma (e.g., trailing comments on argument lines)
            self.skip_comments();
        }
        Ok(exprs)
    }

    fn parse_struct_fields(&mut self) -> ParseResult<(Vec<FieldInit>, Option<Box<Expr>>)> {
        let mut fields = Vec::new();
        let mut rest = None;

        while !self.check(&Token::RBrace) && !self.is_eof() {
            // Skip comments and attributes before field
            while matches!(
                self.current_token(),
                Some(Token::DocComment(_))
                    | Some(Token::LineComment(_) | Token::TildeComment(_) | Token::BlockComment(_))
                    | Some(Token::Hash)
            ) {
                if self.check(&Token::Hash) {
                    // Skip attribute: #[...] or #![...]
                    self.advance();
                    self.consume_if(&Token::Bang); // optional ! for inner attributes
                    if self.consume_if(&Token::LBracket) {
                        let mut depth = 1;
                        while depth > 0 && !self.is_eof() {
                            match self.current_token() {
                                Some(Token::LBracket) => depth += 1,
                                Some(Token::RBracket) => depth -= 1,
                                _ => {}
                            }
                            self.advance();
                        }
                    }
                } else {
                    self.advance();
                }
            }
            if self.check(&Token::RBrace) {
                break;
            }
            if self.consume_if(&Token::DotDot) {
                rest = Some(Box::new(self.parse_expr()?));
                break;
            }

            let name = self.parse_ident()?;
            let value = if self.consume_if(&Token::Colon) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            fields.push(FieldInit { name, value });

            if !self.consume_if(&Token::Comma) {
                break;
            }
        }

        Ok((fields, rest))
    }

    fn is_item_start(&mut self) -> bool {
        match self.current_token() {
            Some(Token::Fn) | Some(Token::Struct) | Some(Token::Enum) | Some(Token::Trait)
            | Some(Token::Impl) | Some(Token::Type) | Some(Token::Mod) | Some(Token::Use)
            | Some(Token::Const) | Some(Token::Static) | Some(Token::Actor) | Some(Token::Pub)
            | Some(Token::Extern) => true,
            // async is only item start if followed by fn (async fn ...)
            // async move { } or async { } are block expressions, not items
            Some(Token::Async) => matches!(self.peek_next(), Some(Token::Fn)),
            _ => false,
        }
    }

    fn is_in_condition(&self) -> bool {
        self.in_condition
    }

    /// Parse an expression in condition context (< is comparison, not generics)
    fn parse_condition(&mut self) -> ParseResult<Expr> {
        let was_in_condition = self.in_condition;
        self.in_condition = true;
        let result = self.parse_expr();
        self.in_condition = was_in_condition;
        result
    }

    // ==========================================
    // Legion Morpheme Parsing
    // ==========================================

    /// Parse Legion operators that follow an expression.
    /// Handles: ⫰ (interference), ⟁ (distribute), ↠ (broadcast),
    ///          ⟀ (gather), ⇢ (consensus), ◉ (resonance)
    fn parse_legion_operator(&mut self, lhs: Expr) -> ParseResult<Expr> {
        match self.current_token() {
            Some(Token::Interfere) => {
                // Interference: query ⫰ field
                self.advance();
                let field = self.parse_expr_bp(15)?; // Higher precedence
                Ok(Expr::LegionInterference {
                    query: Box::new(lhs),
                    field: Box::new(field),
                })
            }
            Some(Token::Distribute) => {
                // Distribute: task ⟁ count
                self.advance();
                let count = self.parse_expr_bp(15)?;
                Ok(Expr::LegionDistribute {
                    task: Box::new(lhs),
                    count: Box::new(count),
                })
            }
            Some(Token::Broadcast) => {
                // Broadcast: signal ↠ target
                self.advance();
                let target = self.parse_expr_bp(15)?;
                Ok(Expr::LegionBroadcast {
                    signal: Box::new(lhs),
                    target: Box::new(target),
                })
            }
            Some(Token::Gather) => {
                // Gather: fragments ⟀ (postfix unary)
                self.advance();
                Ok(Expr::LegionGather {
                    fragments: Box::new(lhs),
                })
            }
            Some(Token::Consensus) => {
                // Consensus: contributions ⇢ (postfix unary)
                self.advance();
                Ok(Expr::LegionConsensus {
                    contributions: Box::new(lhs),
                })
            }
            Some(Token::ConfidenceHigh) => {
                // Resonance: resonance |◉ (postfix unary, dual-purpose token)
                self.advance();
                Ok(Expr::LegionResonance {
                    expr: Box::new(lhs),
                })
            }
            _ => Ok(lhs),
        }
    }

    /// Check if an identifier ends with the Legion field marker ∿
    /// Used to parse variable names like `memory∿`
    fn is_legion_field_ident(&self, name: &str) -> bool {
        name.ends_with('∿')
    }
}

/// Binding power for infix operators.
fn infix_binding_power(op: BinOp) -> (u8, u8) {
    match op {
        BinOp::Or => (1, 2),
        BinOp::And => (3, 4),
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => (5, 6),
        BinOp::BitOr => (7, 8),
        BinOp::BitXor => (9, 10),
        BinOp::BitAnd => (11, 12),
        BinOp::Shl | BinOp::Shr => (13, 14),
        BinOp::Add | BinOp::Sub | BinOp::Concat => (15, 16),
        BinOp::Mul
        | BinOp::Div
        | BinOp::Rem
        | BinOp::MatMul
        | BinOp::Hadamard
        | BinOp::TensorProd
        | BinOp::Convolve => (17, 18),
        BinOp::Pow => (20, 19), // Right associative
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_function() {
        // Simple function with semicolon-terminated statement
        let source = "fn hello(name: str) -> str { return name; }";
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_pipe_chain() {
        let source = "fn main() { let result = data|τ{_ * 2}|φ{_ > 0}|σ; }";
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_async_function() {
        let source = "async fn fetch(url: str) -> Response~ { return client·get(url)|await; }";
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_struct() {
        let source = "struct Point { x: f64, y: f64 }";
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_actor() {
        // Simplified actor without compound assignment
        let source = r#"
            actor Counter {
                state: i64 = 0
                on Increment(n: i64) { return self.state + n; }
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_number_bases() {
        let source = "fn bases() { let a = 42; let b = 0b101010; let c = 0x2A; let d = 0v22; }";
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_labeled_loops() {
        // Test labeled loop with break
        let source = r#"
            fn test() {
                'outer: loop {
                    'inner: while true {
                        break 'outer;
                    }
                }
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        // Test labeled for with continue
        let source2 = r#"
            fn test2() {
                'rows: for i in 0..10 {
                    'cols: for j in 0..10 {
                        if j == 5 { continue 'rows; }
                    }
                }
            }
        "#;
        let mut parser2 = Parser::new(source2);
        let file2 = parser2.parse_file().unwrap();
        assert_eq!(file2.items.len(), 1);
    }

    #[test]
    fn test_parse_inline_asm() {
        let source = r#"
            fn outb(port: u16, value: u8) {
                asm!("out dx, al",
                    in("dx") port,
                    in("al") value,
                    options(nostack));
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        if let Item::Function(func) = &file.items[0].node {
            assert_eq!(func.name.name, "outb");
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_parse_inline_asm_with_outputs() {
        let source = r#"
            fn inb(port: u16) -> u8 {
                let result: u8 = 0;
                asm!("in al, dx",
                    out("al") result,
                    in("dx") port,
                    options(nostack, nomem));
                return result;
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_volatile_read() {
        let source = r#"
            fn read_mmio(addr: *mut u32) -> u32 {
                return volatile read<u32>(addr);
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_volatile_write() {
        let source = r#"
            fn write_mmio(addr: *mut u32, value: u32) {
                volatile write<u32>(addr, value);
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_naked_function() {
        let source = r#"
            naked fn interrupt_handler() {
                asm!("push rax; push rbx; call handler_impl; pop rbx; pop rax; iretq",
                    options(nostack));
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        if let Item::Function(func) = &file.items[0].node {
            assert!(func.attrs.naked, "Function should be naked");
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_parse_packed_struct() {
        let source = r#"
            packed struct GDTEntry {
                limit_low: u16,
                base_low: u16,
                base_middle: u8,
                access: u8,
                granularity: u8,
                base_high: u8,
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        if let Item::Struct(s) = &file.items[0].node {
            assert!(s.attrs.packed, "Struct should be packed");
            assert_eq!(s.name.name, "GDTEntry");
            if let StructFields::Named(fields) = &s.fields {
                assert_eq!(fields.len(), 6);
            } else {
                panic!("Expected named fields");
            }
        } else {
            panic!("Expected struct");
        }
    }

    #[test]
    fn test_parse_no_std_attribute() {
        let source = r#"
            #![no_std]
            #![no_main]

            fn kernel_main() -> ! {
                loop {}
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();

        assert!(file.config.no_std, "Should have no_std");
        assert!(file.config.no_main, "Should have no_main");
        assert_eq!(file.attrs.len(), 2);
    }

    #[test]
    fn test_parse_feature_attribute() {
        let source = r#"
            #![feature(asm, naked_functions)]

            fn main() -> i64 { 0 }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();

        assert_eq!(file.config.features.len(), 2);
        assert!(file.config.features.contains(&"asm".to_string()));
        assert!(file
            .config
            .features
            .contains(&"naked_functions".to_string()));
    }

    #[test]
    fn test_parse_target_attribute() {
        let source = r#"
            #![no_std]
            #![target(arch = "x86_64", os = "none")]

            fn kernel_main() { }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();

        assert!(file.config.no_std);
        let target = file
            .config
            .target
            .as_ref()
            .expect("Should have target config");
        assert_eq!(target.arch, Some("x86_64".to_string()));
        assert_eq!(target.os, Some("none".to_string()));
    }

    #[test]
    fn test_parse_panic_handler() {
        let source = r#"
            #![no_std]

            #[panic_handler]
            fn panic(info: *const PanicInfo) -> ! {
                loop {}
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();

        assert_eq!(file.items.len(), 1);
        if let Item::Function(func) = &file.items[0].node {
            assert!(
                func.attrs.panic_handler,
                "Should have panic_handler attribute"
            );
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_parse_entry_point() {
        let source = r#"
            #![no_std]
            #![no_main]

            #[entry]
            #[no_mangle]
            fn _start() -> ! {
                loop {}
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();

        assert_eq!(file.items.len(), 1);
        if let Item::Function(func) = &file.items[0].node {
            assert!(func.attrs.entry, "Should have entry attribute");
            assert!(func.attrs.no_mangle, "Should have no_mangle attribute");
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_parse_link_section() {
        let source = r#"
            #[link_section = ".text.boot"]
            fn boot_code() { }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();

        assert_eq!(file.items.len(), 1);
        if let Item::Function(func) = &file.items[0].node {
            assert_eq!(func.attrs.link_section, Some(".text.boot".to_string()));
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_parse_linker_config() {
        let source = r#"
            #![no_std]
            #![linker_script = "kernel.ld"]
            #![entry_point = "_start"]
            #![base_address = 0x100000]
            #![stack_size = 0x4000]

            fn kernel_main() { }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();

        let linker = file
            .config
            .linker
            .as_ref()
            .expect("Should have linker config");
        assert_eq!(linker.script, Some("kernel.ld".to_string()));
        assert_eq!(linker.entry_point, Some("_start".to_string()));
        assert_eq!(linker.base_address, Some(0x100000));
        assert_eq!(linker.stack_size, Some(0x4000));
    }

    #[test]
    fn test_parse_interrupt_handler() {
        let source = r#"
            #[interrupt(32)]
            #[naked]
            fn timer_handler() {
                asm!("iretq", options(nostack));
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();

        if let Item::Function(func) = &file.items[0].node {
            assert_eq!(func.attrs.interrupt, Some(32));
            assert!(func.attrs.naked);
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_parse_inline_attributes() {
        let source = r#"
            #[inline]
            fn fast() -> i64 { 0 }

            #[inline(always)]
            fn very_fast() -> i64 { 0 }

            #[inline(never)]
            fn never_inline() -> i64 { 0 }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();

        assert_eq!(file.items.len(), 3);

        if let Item::Function(func) = &file.items[0].node {
            assert_eq!(func.attrs.inline, Some(InlineHint::Hint));
        }
        if let Item::Function(func) = &file.items[1].node {
            assert_eq!(func.attrs.inline, Some(InlineHint::Always));
        }
        if let Item::Function(func) = &file.items[2].node {
            assert_eq!(func.attrs.inline, Some(InlineHint::Never));
        }
    }

    #[test]
    fn test_parse_simd_type() {
        let source = r#"
            fn vec_add(a: simd<f32, 4>, b: simd<f32, 4>) -> simd<f32, 4> {
                return simd.add(a, b);
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        if let Item::Function(func) = &file.items[0].node {
            assert_eq!(func.name.name, "vec_add");
            // Check first parameter type
            if let TypeExpr::Simd { element, lanes } = &func.params[0].ty {
                assert_eq!(*lanes, 4);
                if let TypeExpr::Path(path) = element.as_ref() {
                    assert_eq!(path.segments[0].ident.name, "f32");
                }
            } else {
                panic!("Expected SIMD type");
            }
        } else {
            panic!("Expected function");
        }
    }

    #[test]
    fn test_parse_simd_literal() {
        let source = r#"
            fn make_vec() -> simd<f32, 4> {
                return simd[1.0, 2.0, 3.0, 4.0];
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_simd_intrinsics() {
        let source = r#"
            fn dot_product(a: simd<f32, 4>, b: simd<f32, 4>) -> f32 {
                let prod = simd.mul(a, b);
                return simd.hadd(prod);
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_simd_shuffle() {
        let source = r#"
            fn interleave(a: simd<f32, 4>, b: simd<f32, 4>) -> simd<f32, 4> {
                return simd.shuffle(a, b, [0, 4, 1, 5]);
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_atomic_type() {
        let source = r#"
            struct Counter {
                value: atomic<i64>,
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        if let Item::Struct(s) = &file.items[0].node {
            if let StructFields::Named(fields) = &s.fields {
                if let TypeExpr::Atomic(inner) = &fields[0].ty {
                    if let TypeExpr::Path(path) = inner.as_ref() {
                        assert_eq!(path.segments[0].ident.name, "i64");
                    }
                } else {
                    panic!("Expected atomic type");
                }
            }
        } else {
            panic!("Expected struct");
        }
    }

    #[test]
    fn test_parse_atomic_operations() {
        let source = r#"
            fn increment(ptr: *mut i64) -> i64 {
                return atomic.fetch_add(ptr, 1, SeqCst);
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_atomic_compare_exchange() {
        let source = r#"
            fn cas(ptr: *mut i64, expected: i64, new: i64) -> bool {
                let result = atomic.compare_exchange(ptr, expected, new, AcqRel, Relaxed);
                return result;
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_atomic_fence() {
        let source = r#"
            fn memory_barrier() {
                atomic.fence(SeqCst);
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_parse_derive_macro() {
        let source = r#"
            #[derive(Debug, Clone, Component)]
            struct Position {
                x: f32,
                y: f32,
                z: f32,
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        if let Item::Struct(s) = &file.items[0].node {
            assert_eq!(s.attrs.derives.len(), 3);
            assert!(matches!(s.attrs.derives[0], DeriveTrait::Debug));
            assert!(matches!(s.attrs.derives[1], DeriveTrait::Clone));
            assert!(matches!(s.attrs.derives[2], DeriveTrait::Component));
        } else {
            panic!("Expected struct");
        }
    }

    #[test]
    fn test_parse_repr_c_struct() {
        let source = r#"
            #[repr(C)]
            struct FFIStruct {
                field: i32,
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        if let Item::Struct(s) = &file.items[0].node {
            assert_eq!(s.attrs.repr, Some(StructRepr::C));
        } else {
            panic!("Expected struct");
        }
    }

    #[test]
    fn test_parse_allocator_trait() {
        let source = r#"
            trait Allocator {
                type Error;

                fn allocate(size: usize, align: usize) -> *mut u8;
                fn deallocate(ptr: *mut u8, size: usize, align: usize);
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        if let Item::Trait(t) = &file.items[0].node {
            assert_eq!(t.name.name, "Allocator");
            assert_eq!(t.items.len(), 3); // associated type + 2 methods
            assert!(matches!(t.items[0], TraitItem::Type { .. }));
        } else {
            panic!("Expected trait");
        }
    }

    #[test]
    fn test_parse_where_clause() {
        let source = r#"
            fn alloc_array<T, A>(allocator: &mut A, count: usize) -> *mut T
            where
                A: Allocator,
            {
                return allocator.allocate(count, 8);
            }
        "#;
        let mut parser = Parser::new(source);
        let file = parser.parse_file().unwrap();
        assert_eq!(file.items.len(), 1);

        if let Item::Function(func) = &file.items[0].node {
            assert!(func.where_clause.is_some());
            let wc = func.where_clause.as_ref().unwrap();
            assert_eq!(wc.predicates.len(), 1);
        } else {
            panic!("Expected function");
        }
    }
}
