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
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let current = lexer.next_token();
        Self { lexer, current, in_condition: false }
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
            while matches!(self.current_token(), Some(Token::LineComment(_) | Token::DocComment(_))) {
                self.advance();
            }
            if self.is_eof() {
                break;
            }
            items.push(self.parse_item()?);
        }
        Ok(SourceFile { attrs, config, items })
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

    /// Parse an outer attribute: `#[name]` or `#[name(args)]`
    fn parse_outer_attribute(&mut self) -> ParseResult<Attribute> {
        self.expect(Token::Hash)?;
        self.expect(Token::LBracket)?;

        let name = self.parse_attr_name()?;
        let args = self.parse_attr_args()?;

        self.expect(Token::RBracket)?;

        Ok(Attribute {
            name,
            args,
            is_inner: false,
        })
    }

    /// Parse an attribute name (identifier or keyword used as attribute).
    fn parse_attr_name(&mut self) -> ParseResult<Ident> {
        let span = self.current_span();
        match self.current_token().cloned() {
            Some(Token::Ident(name)) => {
                self.advance();
                Ok(Ident { name, evidentiality: None, span })
            }
            // Handle keywords that can be used as attribute names
            Some(Token::Naked) => {
                self.advance();
                Ok(Ident { name: "naked".to_string(), evidentiality: None, span })
            }
            Some(Token::Unsafe) => {
                self.advance();
                Ok(Ident { name: "unsafe".to_string(), evidentiality: None, span })
            }
            Some(Token::Asm) => {
                self.advance();
                Ok(Ident { name: "asm".to_string(), evidentiality: None, span })
            }
            Some(Token::Volatile) => {
                self.advance();
                Ok(Ident { name: "volatile".to_string(), evidentiality: None, span })
            }
            Some(Token::Derive) => {
                self.advance();
                Ok(Ident { name: "derive".to_string(), evidentiality: None, span })
            }
            Some(Token::Simd) => {
                self.advance();
                Ok(Ident { name: "simd".to_string(), evidentiality: None, span })
            }
            Some(Token::Atomic) => {
                self.advance();
                Ok(Ident { name: "atomic".to_string(), evidentiality: None, span })
            }
            Some(t) => Err(ParseError::UnexpectedToken {
                expected: "attribute name".to_string(),
                found: t,
                span,
            }),
            None => Err(ParseError::UnexpectedEof),
        }
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
                let ident = Ident { name: "asm".to_string(), evidentiality: None, span };
                self.parse_attr_arg_after_ident(ident)
            }
            Some(Token::Volatile) => {
                let span = self.current_span();
                self.advance();
                let ident = Ident { name: "volatile".to_string(), evidentiality: None, span };
                self.parse_attr_arg_after_ident(ident)
            }
            Some(Token::Naked) => {
                let span = self.current_span();
                self.advance();
                let ident = Ident { name: "naked".to_string(), evidentiality: None, span };
                self.parse_attr_arg_after_ident(ident)
            }
            Some(Token::Packed) => {
                let span = self.current_span();
                self.advance();
                let ident = Ident { name: "packed".to_string(), evidentiality: None, span };
                self.parse_attr_arg_after_ident(ident)
            }
            Some(Token::Unsafe) => {
                let span = self.current_span();
                self.advance();
                let ident = Ident { name: "unsafe".to_string(), evidentiality: None, span };
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
            NumBase::Binary => (s.strip_prefix("0b").or(s.strip_prefix("0B")).unwrap_or(s), 2),
            NumBase::Octal => (s.strip_prefix("0o").or(s.strip_prefix("0O")).unwrap_or(s), 8),
            NumBase::Decimal => (s, 10),
            NumBase::Hex => (s.strip_prefix("0x").or(s.strip_prefix("0X")).unwrap_or(s), 16),
            NumBase::Vigesimal => (s.strip_prefix("0v").or(s.strip_prefix("0V")).unwrap_or(s), 20),
            NumBase::Duodecimal => (s.strip_prefix("0d").or(s.strip_prefix("0D")).unwrap_or(s), 12),
            NumBase::Sexagesimal => (s.strip_prefix("0s").or(s.strip_prefix("0S")).unwrap_or(s), 60),
            NumBase::Explicit(r) => (s, r as u32),
        };
        // Remove underscores (numeric separators) and parse
        let clean: String = stripped.chars().filter(|c| *c != '_').collect();
        u64::from_str_radix(&clean, radix).unwrap_or(0)
    }

    // === Token utilities ===

    fn current_token(&self) -> Option<&Token> {
        self.current.as_ref().map(|(t, _)| t)
    }

    fn current_span(&self) -> Span {
        self.current.as_ref().map(|(_, s)| *s).unwrap_or_default()
    }

    fn advance(&mut self) -> Option<(Token, Span)> {
        let prev = self.current.take();
        self.current = self.lexer.next_token();
        prev
    }

    fn is_eof(&self) -> bool {
        self.current.is_none()
    }

    fn expect(&mut self, expected: Token) -> ParseResult<Span> {
        match &self.current {
            Some((token, span)) if std::mem::discriminant(token) == std::mem::discriminant(&expected) => {
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

    fn check(&self, expected: &Token) -> bool {
        matches!(&self.current, Some((token, _)) if std::mem::discriminant(token) == std::mem::discriminant(expected))
    }

    /// Peek at the next token (after current) without consuming anything.
    fn peek_next(&mut self) -> Option<&Token> {
        self.lexer.peek().map(|(t, _)| t)
    }

    fn consume_if(&mut self, expected: &Token) -> bool {
        if self.check(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Skip any comments
    fn skip_comments(&mut self) {
        while matches!(self.current_token(), Some(Token::LineComment(_)) | Some(Token::DocComment(_))) {
            self.advance();
        }
    }

    // === Item parsing ===

    fn parse_item(&mut self) -> ParseResult<Spanned<Item>> {
        let start_span = self.current_span();

        // Collect outer attributes (#[...])
        let mut outer_attrs = Vec::new();
        while self.check(&Token::Hash) {
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
            Some(Token::Enum) => {
                Item::Enum(self.parse_enum(visibility)?)
            }
            Some(Token::Trait) => {
                Item::Trait(self.parse_trait(visibility)?)
            }
            Some(Token::Impl) => {
                Item::Impl(self.parse_impl()?)
            }
            Some(Token::Type) => {
                Item::TypeAlias(self.parse_type_alias(visibility)?)
            }
            Some(Token::Mod) => {
                Item::Module(self.parse_module(visibility)?)
            }
            Some(Token::Use) => {
                Item::Use(self.parse_use(visibility)?)
            }
            Some(Token::Const) => {
                Item::Const(self.parse_const(visibility)?)
            }
            Some(Token::Static) => {
                Item::Static(self.parse_static(visibility)?)
            }
            Some(Token::Actor) => {
                Item::Actor(self.parse_actor(visibility)?)
            }
            Some(Token::Extern) => {
                Item::ExternBlock(self.parse_extern_block()?)
            }
            Some(Token::Naked) => {
                // naked fn -> function with naked attribute
                Item::Function(self.parse_function_with_attrs(visibility, outer_attrs)?)
            }
            Some(Token::Packed) => {
                // packed struct -> struct with packed attribute
                Item::Struct(self.parse_struct_with_attrs(visibility, outer_attrs)?)
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

    fn parse_visibility(&mut self) -> ParseResult<Visibility> {
        if self.consume_if(&Token::Pub) {
            Ok(Visibility::Public)
        } else {
            Ok(Visibility::Private)
        }
    }

    fn parse_function(&mut self, visibility: Visibility) -> ParseResult<Function> {
        self.parse_function_with_attrs(visibility, Vec::new())
    }

    fn parse_function_with_attrs(&mut self, visibility: Visibility, outer_attrs: Vec<Attribute>) -> ParseResult<Function> {
        // Parse function attributes from outer attributes
        let mut attrs = self.process_function_attrs(&outer_attrs);

        // Check for naked keyword before fn
        if self.consume_if(&Token::Naked) {
            attrs.naked = true;
        }

        let is_async = self.consume_if(&Token::Async);
        self.expect(Token::Fn)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics_opt()?;

        self.expect(Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(Token::RParen)?;

        let return_type = if self.consume_if(&Token::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let where_clause = self.parse_where_clause_opt()?;

        let body = if self.check(&Token::LBrace) {
            Some(self.parse_block()?)
        } else {
            self.expect(Token::Semi)?;
            None
        };

        Ok(Function {
            visibility,
            is_async,
            attrs,
            name,
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
                        if let Some(AttrArg::Literal(Literal::Int { value, base, .. })) = args.first() {
                            let num = Self::parse_int_value(value, *base) as u32;
                            func_attrs.interrupt = Some(num);
                        }
                    }
                }
                "align" => {
                    if let Some(AttrArgs::Paren(args)) = &attr.args {
                        if let Some(AttrArg::Literal(Literal::Int { value, base, .. })) = args.first() {
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

    fn parse_struct_with_attrs(&mut self, visibility: Visibility, outer_attrs: Vec<Attribute>) -> ParseResult<StructDef> {
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
                                    if let Some(AttrArg::Literal(Literal::Int { value, .. })) = align_args.first() {
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
        let generics = self.parse_generics_opt()?;

        let fields = if self.check(&Token::LBrace) {
            self.expect(Token::LBrace)?;
            let fields = self.parse_field_defs()?;
            self.expect(Token::RBrace)?;
            StructFields::Named(fields)
        } else if self.check(&Token::LParen) {
            self.expect(Token::LParen)?;
            let types = self.parse_type_list()?;
            self.expect(Token::RParen)?;
            self.expect(Token::Semi)?;
            StructFields::Tuple(types)
        } else {
            self.expect(Token::Semi)?;
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
            variants.push(self.parse_enum_variant()?);
            if !self.consume_if(&Token::Comma) {
                break;
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
            let types = self.parse_type_list()?;
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
            Some(Token::Fn) | Some(Token::Async) => {
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
                self.advance();
                let name = self.parse_ident()?;
                self.expect(Token::Colon)?;
                let ty = self.parse_type()?;
                self.expect(Token::Semi)?;
                Ok(TraitItem::Const { name, ty })
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

        self.expect(Token::LBrace)?;
        let mut items = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_eof() {
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
        let visibility = self.parse_visibility()?;

        match self.current_token() {
            Some(Token::Fn) | Some(Token::Async) => {
                Ok(ImplItem::Function(self.parse_function(visibility)?))
            }
            Some(Token::Type) => {
                Ok(ImplItem::Type(self.parse_type_alias(visibility)?))
            }
            Some(Token::Const) => {
                Ok(ImplItem::Const(self.parse_const(visibility)?))
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
        self.expect(Token::Eq)?;
        let ty = self.parse_type()?;
        self.expect(Token::Semi)?;

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
                items.push(self.parse_item()?);
            }
            self.expect(Token::RBrace)?;
            Some(items)
        } else {
            self.expect(Token::Semi)?;
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
        self.expect(Token::Semi)?;

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
                trees.push(self.parse_use_tree()?);
                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
            self.expect(Token::RBrace)?;
            return Ok(UseTree::Group(trees));
        }

        let name = self.parse_ident()?;

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
        self.expect(Token::Semi)?;

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
        self.expect(Token::Semi)?;

        Ok(StaticDef {
            visibility,
            mutable,
            name,
            ty,
            value,
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
                if self.consume_if(&Token::Eq) {
                    let _ = self.parse_expr()?;
                }

                if !self.check(&Token::RBrace) && !self.check(&Token::On) {
                    self.consume_if(&Token::Comma);
                }

                state.push(FieldDef {
                    visibility: vis,
                    name: field_name,
                    ty,
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
                    items.push(ExternItem::Function(self.parse_extern_function(visibility)?));
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

    fn parse_type(&mut self) -> ParseResult<TypeExpr> {
        let base = self.parse_type_base()?;

        // Check for evidentiality suffix
        if let Some(ev) = self.parse_evidentiality_opt() {
            return Ok(TypeExpr::Evidential {
                inner: Box::new(base),
                evidentiality: ev,
            });
        }

        Ok(base)
    }

    fn parse_type_base(&mut self) -> ParseResult<TypeExpr> {
        match self.current_token() {
            Some(Token::Amp) => {
                self.advance();
                let mutable = self.consume_if(&Token::Mut);
                let inner = self.parse_type()?;
                Ok(TypeExpr::Reference {
                    mutable,
                    inner: Box::new(inner),
                })
            }
            Some(Token::Star) => {
                self.advance();
                let mutable = if self.consume_if(&Token::Const) {
                    false
                } else {
                    self.expect(Token::Mut)?;
                    true
                };
                let inner = self.parse_type()?;
                Ok(TypeExpr::Pointer {
                    mutable,
                    inner: Box::new(inner),
                })
            }
            Some(Token::LBracket) => {
                self.advance();
                let element = self.parse_type()?;
                if self.consume_if(&Token::Semi) {
                    let size = self.parse_expr()?;
                    self.expect(Token::RBracket)?;
                    Ok(TypeExpr::Array {
                        element: Box::new(element),
                        size: Box::new(size),
                    })
                } else {
                    self.expect(Token::RBracket)?;
                    Ok(TypeExpr::Slice(Box::new(element)))
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
            Some(Token::Bang) => {
                self.advance();
                Ok(TypeExpr::Never)
            }
            Some(Token::Underscore) => {
                self.advance();
                Ok(TypeExpr::Infer)
            }
            Some(Token::Simd) => {
                self.advance();
                self.expect(Token::Lt)?;
                let element = self.parse_type()?;
                self.expect(Token::Comma)?;
                let lanes = match self.current_token() {
                    Some(Token::IntLit(s)) => {
                        let n = s.parse::<u8>().map_err(|_| ParseError::Custom("invalid lane count".to_string()))?;
                        self.advance();
                        n
                    }
                    _ => return Err(ParseError::Custom("expected lane count".to_string())),
                };
                self.expect(Token::Gt)?;
                Ok(TypeExpr::Simd {
                    element: Box::new(element),
                    lanes,
                })
            }
            Some(Token::Atomic) => {
                self.advance();
                self.expect(Token::Lt)?;
                let inner = self.parse_type()?;
                self.expect(Token::Gt)?;
                Ok(TypeExpr::Atomic(Box::new(inner)))
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

        while self.consume_if(&Token::ColonColon) || self.consume_if(&Token::MiddleDot) {
            segments.push(self.parse_path_segment()?);
        }

        Ok(TypePath { segments })
    }

    fn parse_path_segment(&mut self) -> ParseResult<PathSegment> {
        let ident = self.parse_ident()?;

        // Don't parse generics in condition context (< is comparison, not generics)
        let generics = if !self.is_in_condition() && self.consume_if(&Token::Lt) {
            let types = self.parse_type_list()?;
            self.expect(Token::Gt)?;
            Some(types)
        } else {
            None
        };

        Ok(PathSegment { ident, generics })
    }

    fn parse_type_list(&mut self) -> ParseResult<Vec<TypeExpr>> {
        let mut types = Vec::new();
        if !self.check(&Token::RParen) && !self.check(&Token::RBracket) && !self.check(&Token::Gt) {
            types.push(self.parse_type()?);
            while self.consume_if(&Token::Comma) {
                if self.check(&Token::RParen) || self.check(&Token::RBracket) || self.check(&Token::Gt) {
                    break;
                }
                types.push(self.parse_type()?);
            }
        }
        Ok(types)
    }

    fn parse_type_bounds(&mut self) -> ParseResult<Vec<TypeExpr>> {
        let mut bounds = Vec::new();
        bounds.push(self.parse_type()?);
        while self.consume_if(&Token::Plus) {
            bounds.push(self.parse_type()?);
        }
        Ok(bounds)
    }

    // === Expression parsing (Pratt parser) ===

    fn parse_expr(&mut self) -> ParseResult<Expr> {
        let lhs = self.parse_expr_bp(0)?;

        // Check for assignment: expr = value
        if self.consume_if(&Token::Eq) {
            let value = self.parse_expr()?;
            return Ok(Expr::Assign {
                target: Box::new(lhs),
                value: Box::new(value),
            });
        }

        Ok(lhs)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> ParseResult<Expr> {
        let mut lhs = self.parse_prefix_expr()?;

        loop {
            // Check for pipe operator
            if self.check(&Token::Pipe) {
                lhs = self.parse_pipe_chain(lhs)?;
                continue;
            }

            // Check for binary operators
            let op = match self.current_token() {
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
                _ => break,
            };

            let (l_bp, r_bp) = infix_binding_power(op);
            if l_bp < min_bp {
                break;
            }

            self.advance();
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
            _ => self.parse_postfix_expr(),
        }
    }

    fn parse_postfix_expr(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_primary_expr()?;

        loop {
            match self.current_token() {
                Some(Token::LParen) => {
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
                    let index = self.parse_expr()?;
                    self.expect(Token::RBracket)?;
                    expr = Expr::Index {
                        expr: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                Some(Token::Dot) => {
                    self.advance();
                    let field = self.parse_ident()?;
                    if self.check(&Token::LParen) {
                        self.advance();
                        let args = self.parse_expr_list()?;
                        self.expect(Token::RParen)?;
                        expr = Expr::MethodCall {
                            receiver: Box::new(expr),
                            method: field,
                            args,
                        };
                    } else {
                        expr = Expr::Field {
                            expr: Box::new(expr),
                            field,
                        };
                    }
                }
                Some(Token::Question) => {
                    self.advance();
                    expr = Expr::Try(Box::new(expr));
                }
                Some(Token::Bang) | Some(Token::Tilde) | Some(Token::Interrobang) => {
                    if let Some(ev) = self.parse_evidentiality_opt() {
                        expr = Expr::Evidential {
                            expr: Box::new(expr),
                            evidentiality: ev,
                        };
                    } else {
                        break;
                    }
                }
                Some(Token::Hourglass) => {
                    self.advance();
                    expr = Expr::Await(Box::new(expr));
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
            Some(Token::CharLit(c)) => {
                self.advance();
                Ok(Expr::Literal(Literal::Char(c)))
            }
            Some(Token::True) => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(true)))
            }
            Some(Token::False) => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(false)))
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
                let exprs = self.parse_expr_list()?;
                self.expect(Token::RBracket)?;
                Ok(Expr::Array(exprs))
            }
            Some(Token::LBrace) => {
                // Could be block or closure
                self.parse_block_or_closure()
            }
            Some(Token::If) => self.parse_if_expr(),
            Some(Token::Match) => self.parse_match_expr(),
            Some(Token::Loop) => {
                self.advance();
                let body = self.parse_block()?;
                Ok(Expr::Loop(body))
            }
            Some(Token::While) => {
                self.advance();
                let condition = self.parse_condition()?;
                let body = self.parse_block()?;
                Ok(Expr::While {
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
                    pattern,
                    iter: Box::new(iter),
                    body,
                })
            }
            Some(Token::Return) => {
                self.advance();
                let value = if self.check(&Token::Semi) || self.check(&Token::RBrace) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                Ok(Expr::Return(value))
            }
            Some(Token::Break) => {
                self.advance();
                let value = if self.check(&Token::Semi) || self.check(&Token::RBrace) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                Ok(Expr::Break(value))
            }
            Some(Token::Continue) => {
                self.advance();
                Ok(Expr::Continue)
            }
            // Morphemes as standalone expressions
            Some(Token::Tau) | Some(Token::Phi) | Some(Token::Sigma) |
            Some(Token::Rho) | Some(Token::Lambda) | Some(Token::Pi) => {
                let kind = self.parse_morpheme_kind()?;
                if self.check(&Token::LBrace) {
                    self.advance();
                    let body = self.parse_expr()?;
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
                                    span: Span::default(),
                                },
                                generics: None,
                            }],
                        })),
                    })
                }
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
                            span,
                        },
                        generics: None,
                    }],
                }))
            }
            Some(Token::Ident(_)) => {
                let path = self.parse_type_path()?;

                // Check for struct literal
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
                    self.expect(Token::Gt)?;
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
                    self.expect(Token::Gt)?;
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
                                        let n = s.parse::<u8>().map_err(|_| ParseError::Custom("invalid lane count".to_string()))?;
                                        self.advance();
                                        n
                                    }
                                    _ => return Err(ParseError::Custom("expected lane count".to_string())),
                                };
                                self.expect(Token::RParen)?;
                                Ok(Expr::SimdSplat { value: Box::new(value), lanes })
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
                                            let n = s.parse::<u8>().map_err(|_| ParseError::Custom("invalid index".to_string()))?;
                                            indices.push(n);
                                            self.advance();
                                        }
                                        _ => return Err(ParseError::Custom("expected index".to_string())),
                                    }
                                    if !self.consume_if(&Token::Comma) {
                                        break;
                                    }
                                }
                                self.expect(Token::RBracket)?;
                                self.expect(Token::RParen)?;
                                Ok(Expr::SimdShuffle { a: Box::new(a), b: Box::new(b), indices })
                            }
                            "extract" => {
                                let vector = self.parse_expr()?;
                                self.expect(Token::Comma)?;
                                let index = match self.current_token() {
                                    Some(Token::IntLit(s)) => {
                                        let n = s.parse::<u8>().map_err(|_| ParseError::Custom("invalid index".to_string()))?;
                                        self.advance();
                                        n
                                    }
                                    _ => return Err(ParseError::Custom("expected index".to_string())),
                                };
                                self.expect(Token::RParen)?;
                                Ok(Expr::SimdExtract { vector: Box::new(vector), index })
                            }
                            "insert" => {
                                let vector = self.parse_expr()?;
                                self.expect(Token::Comma)?;
                                let index = match self.current_token() {
                                    Some(Token::IntLit(s)) => {
                                        let n = s.parse::<u8>().map_err(|_| ParseError::Custom("invalid index".to_string()))?;
                                        self.advance();
                                        n
                                    }
                                    _ => return Err(ParseError::Custom("expected index".to_string())),
                                };
                                self.expect(Token::Comma)?;
                                let value = self.parse_expr()?;
                                self.expect(Token::RParen)?;
                                Ok(Expr::SimdInsert { vector: Box::new(vector), index, value: Box::new(value) })
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
            _ => Err(ParseError::Custom(format!("unknown SIMD operation: {}", name))),
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
            _ => Err(ParseError::Custom(format!("unknown atomic operation: {}", name))),
        }
    }

    fn parse_memory_ordering(&mut self) -> ParseResult<MemoryOrdering> {
        match self.current_token() {
            Some(Token::Ident(name)) => {
                let ordering = match name.as_str() {
                    "Relaxed" => MemoryOrdering::Relaxed,
                    "Acquire" => MemoryOrdering::Acquire,
                    "Release" => MemoryOrdering::Release,
                    "AcqRel" => MemoryOrdering::AcqRel,
                    "SeqCst" => MemoryOrdering::SeqCst,
                    _ => return Err(ParseError::Custom("expected memory ordering (Relaxed, Acquire, Release, AcqRel, SeqCst)".to_string())),
                };
                self.advance();
                Ok(ordering)
            }
            _ => Err(ParseError::Custom("expected memory ordering".to_string())),
        }
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
                let body = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Transform(Box::new(body)))
            }
            Some(Token::Phi) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let body = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Filter(Box::new(body)))
            }
            Some(Token::Sigma) => {
                self.advance();
                let field = if self.consume_if(&Token::Dot) {
                    Some(self.parse_ident()?)
                } else {
                    None
                };
                Ok(PipeOp::Sort(field))
            }
            Some(Token::Rho) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let body = self.parse_expr()?;
                self.expect(Token::RBrace)?;
                Ok(PipeOp::Reduce(Box::new(body)))
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
            Some(Token::Ident(_)) => {
                let name = self.parse_ident()?;
                let args = if self.check(&Token::LParen) {
                    self.advance();
                    let args = self.parse_expr_list()?;
                    self.expect(Token::RParen)?;
                    args
                } else {
                    vec![]
                };
                Ok(PipeOp::Method { name, args })
            }
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "pipe operation".to_string(),
                found: token.clone(),
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
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
                body: Box::new(body),
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

            if self.is_item_start() {
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

    fn parse_block(&mut self) -> ParseResult<Block> {
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
        self.expect(Token::Semi)?;

        Ok(Stmt::Let { pattern, ty, init })
    }

    fn parse_if_expr(&mut self) -> ParseResult<Expr> {
        self.expect(Token::If)?;
        let condition = self.parse_condition()?;
        let then_branch = self.parse_block()?;
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
        let expr = self.parse_expr()?;
        self.expect(Token::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_eof() {
            let pattern = self.parse_pattern()?;
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
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }

        self.expect(Token::RBrace)?;

        Ok(Expr::Match {
            expr: Box::new(expr),
            arms,
        })
    }

    // === Pattern parsing ===

    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
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
                let name = self.parse_ident()?;
                let evidentiality = self.parse_evidentiality_opt();
                Ok(Pattern::Ident {
                    mutable: true,
                    name,
                    evidentiality,
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
            Some(Token::IntLit(_)) | Some(Token::StringLit(_)) |
            Some(Token::CharLit(_)) | Some(Token::True) | Some(Token::False) => {
                let lit = self.parse_literal()?;
                Ok(Pattern::Literal(lit))
            }
            Some(Token::Ident(_)) => {
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
            _ => Err(ParseError::Custom("expected literal".to_string())),
        }
    }

    // === Helpers ===

    fn parse_ident(&mut self) -> ParseResult<Ident> {
        match self.current.take() {
            Some((Token::Ident(name), span)) => {
                self.current = self.lexer.next_token();
                Ok(Ident {
                    name,
                    evidentiality: None,
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
        match self.current_token() {
            Some(Token::Bang) => {
                self.advance();
                Some(Evidentiality::Known)
            }
            Some(Token::Question) => {
                self.advance();
                Some(Evidentiality::Uncertain)
            }
            Some(Token::Tilde) => {
                self.advance();
                Some(Evidentiality::Reported)
            }
            Some(Token::Interrobang) => {
                self.advance();
                Some(Evidentiality::Paradox)
            }
            _ => None,
        }
    }

    fn parse_generics_opt(&mut self) -> ParseResult<Option<Generics>> {
        if !self.consume_if(&Token::Lt) {
            return Ok(None);
        }

        let mut params = Vec::new();
        while !self.check(&Token::Gt) && !self.is_eof() {
            // Simple type parameter for now
            let name = self.parse_ident()?;
            let evidentiality = self.parse_evidentiality_opt();
            let bounds = if self.consume_if(&Token::Colon) {
                self.parse_type_bounds()?
            } else {
                vec![]
            };
            params.push(GenericParam::Type {
                name,
                bounds,
                evidentiality,
            });
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        self.expect(Token::Gt)?;

        Ok(Some(Generics { params }))
    }

    fn parse_where_clause_opt(&mut self) -> ParseResult<Option<WhereClause>> {
        if !self.consume_if(&Token::Where) {
            return Ok(None);
        }

        let mut predicates = Vec::new();
        loop {
            let ty = self.parse_type()?;
            self.expect(Token::Colon)?;
            let bounds = self.parse_type_bounds()?;
            predicates.push(WherePredicate { ty, bounds });
            if !self.consume_if(&Token::Comma) {
                break;
            }
            // Stop if we hit a brace
            if self.check(&Token::LBrace) {
                break;
            }
        }

        Ok(Some(WhereClause { predicates }))
    }

    fn parse_params(&mut self) -> ParseResult<Vec<Param>> {
        let mut params = Vec::new();
        while !self.check(&Token::RParen) && !self.is_eof() {
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

    fn parse_field_defs(&mut self) -> ParseResult<Vec<FieldDef>> {
        let mut fields = Vec::new();
        while !self.check(&Token::RBrace) && !self.is_eof() {
            let visibility = self.parse_visibility()?;
            let name = self.parse_ident()?;
            self.expect(Token::Colon)?;
            let ty = self.parse_type()?;
            fields.push(FieldDef {
                visibility,
                name,
                ty,
            });
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        Ok(fields)
    }

    fn parse_expr_list(&mut self) -> ParseResult<Vec<Expr>> {
        let mut exprs = Vec::new();
        while !self.check(&Token::RParen) && !self.check(&Token::RBracket) && !self.is_eof() {
            exprs.push(self.parse_expr()?);
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }
        Ok(exprs)
    }

    fn parse_struct_fields(&mut self) -> ParseResult<(Vec<FieldInit>, Option<Box<Expr>>)> {
        let mut fields = Vec::new();
        let mut rest = None;

        while !self.check(&Token::RBrace) && !self.is_eof() {
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

    fn is_item_start(&self) -> bool {
        matches!(
            self.current_token(),
            Some(Token::Fn) | Some(Token::Async) | Some(Token::Struct) |
            Some(Token::Enum) | Some(Token::Trait) | Some(Token::Impl) |
            Some(Token::Type) | Some(Token::Mod) | Some(Token::Use) |
            Some(Token::Const) | Some(Token::Static) | Some(Token::Actor) |
            Some(Token::Pub)
        )
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
        BinOp::Mul | BinOp::Div | BinOp::Rem => (17, 18),
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
        let source = "async fn fetch(url: str) -> Response~ { return http·get(url)|await; }";
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
        assert!(file.config.features.contains(&"naked_functions".to_string()));
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
        let target = file.config.target.as_ref().expect("Should have target config");
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
            assert!(func.attrs.panic_handler, "Should have panic_handler attribute");
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

        let linker = file.config.linker.as_ref().expect("Should have linker config");
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
