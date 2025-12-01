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
        Ok(SourceFile { items })
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
        let visibility = self.parse_visibility()?;

        let item = match self.current_token() {
            Some(Token::Fn) | Some(Token::Async) => {
                Item::Function(self.parse_function(visibility)?)
            }
            Some(Token::Struct) => {
                Item::Struct(self.parse_struct(visibility)?)
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
            name,
            generics,
            params,
            return_type,
            where_clause,
            body,
        })
    }

    fn parse_struct(&mut self, visibility: Visibility) -> ParseResult<StructDef> {
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
            name,
            generics,
            fields,
        })
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
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: token,
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
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
}
