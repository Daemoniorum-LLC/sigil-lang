//! Sigil Parser - Produces AST from tokens

use crate::lexer::{Lexer, Token};

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Ident(String),
    Binary { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    Unary { op: UnaryOp, expr: Box<Expr> },
    Call { func: Box<Expr>, args: Vec<Expr> },
    If { cond: Box<Expr>, then_: Box<Expr>, else_: Option<Box<Expr>> },
    Block(Vec<Stmt>),
    Match { expr: Box<Expr>, arms: Vec<MatchArm> },
    FieldAccess { expr: Box<Expr>, field: String },
    StructLit { name: String, fields: Vec<(String, Expr)> },
    Array(Vec<Expr>),
    Index { expr: Box<Expr>, index: Box<Expr> },
    // Morpheme operations
    Morpheme { op: MorphOp, expr: Box<Expr>, closure: Option<Box<Expr>> },
}

#[derive(Debug, Clone, Copy)]
pub enum MorphOp {
    Tau,    // τ - transform/map
    Phi,    // φ - filter
    Sigma,  // Σ - sum
    Pi,     // Π - product
    Mu,     // μ - mean
    Alpha,  // α - first
    Omega,  // ω - last
    Lambda, // λ - length
    Sort,   // σ - sort
    Rho,    // ρ - reduce
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Int(i64),
    Ident(String),
    Variant { name: String, fields: Option<Vec<String>> },
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, NotEq, Lt, LtEq, Gt, GtEq,
    And, Or,
    Concat, // ++ string concatenation
    Assign, // = variable reassignment
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg, Not,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let { name: String, ty: Option<String>, value: Expr, mutable: bool },
    Expr(Expr),
    Return(Option<Expr>),
    While { cond: Expr, body: Expr },
}

#[derive(Debug, Clone)]
pub enum Item {
    Function {
        name: String,
        params: Vec<(String, String)>,
        ret_ty: Option<String>,
        body: Vec<Stmt>,
    },
    Struct {
        name: String,
        fields: Vec<(String, String)>,
    },
    Enum {
        name: String,
        variants: Vec<(String, Option<Vec<String>>)>,
    },
    Impl {
        name: String,
        methods: Vec<Item>,
    },
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let current = lexer.next_token();
        Self { lexer, current }
    }

    fn advance(&mut self) -> Token {
        let prev = std::mem::replace(&mut self.current, self.lexer.next_token());
        prev
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        if std::mem::discriminant(&self.current) == std::mem::discriminant(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(format!("Expected {:?}, got {:?}", expected, self.current))
        }
    }

    fn check(&self, token: &Token) -> bool {
        std::mem::discriminant(&self.current) == std::mem::discriminant(token)
    }

    pub fn parse(&mut self) -> Result<Vec<Item>, String> {
        let mut items = Vec::new();
        while !self.check(&Token::Eof) {
            items.push(self.parse_item()?);
        }
        Ok(items)
    }

    fn parse_item(&mut self) -> Result<Item, String> {
        match &self.current {
            Token::Rite => self.parse_function(),
            Token::Sigil => self.parse_struct(),
            Token::Enum => self.parse_enum(),
            Token::Impl => self.parse_impl(),
            _ => Err(format!("Expected item, got {:?}", self.current)),
        }
    }

    fn parse_function(&mut self) -> Result<Item, String> {
        self.advance(); // consume 'rite'

        let name = match self.advance() {
            Token::Ident(s) => s,
            t => return Err(format!("Expected function name, got {:?}", t)),
        };

        self.expect(Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(Token::RParen)?;

        let ret_ty = if self.check(&Token::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(Token::LBrace)?;
        let body = self.parse_block_contents()?;
        self.expect(Token::RBrace)?;

        Ok(Item::Function { name, params, ret_ty, body })
    }

    fn parse_params(&mut self) -> Result<Vec<(String, String)>, String> {
        let mut params = Vec::new();

        // Handle 'self' parameter
        if self.check(&Token::Self_) {
            self.advance();
            params.push(("self".to_string(), "Self".to_string()));
            if self.check(&Token::Comma) {
                self.advance();
            }
        }

        while !self.check(&Token::RParen) {
            let name = match self.advance() {
                Token::Ident(s) => s,
                _ => break,
            };
            self.expect(Token::Colon)?;
            let ty = self.parse_type()?;
            params.push((name, ty));

            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }
        Ok(params)
    }

    fn parse_type(&mut self) -> Result<String, String> {
        match self.advance() {
            Token::Ident(s) => Ok(s),
            t => Err(format!("Expected type, got {:?}", t)),
        }
    }

    fn parse_struct(&mut self) -> Result<Item, String> {
        self.advance(); // consume 'sigil'

        let name = match self.advance() {
            Token::Ident(s) => s,
            t => return Err(format!("Expected struct name, got {:?}", t)),
        };

        self.expect(Token::LBrace)?;
        let fields = self.parse_struct_fields()?;
        self.expect(Token::RBrace)?;

        Ok(Item::Struct { name, fields })
    }

    fn parse_struct_fields(&mut self) -> Result<Vec<(String, String)>, String> {
        let mut fields = Vec::new();
        while !self.check(&Token::RBrace) {
            let name = match self.advance() {
                Token::Ident(s) => s,
                _ => break,
            };
            self.expect(Token::Colon)?;
            let ty = self.parse_type()?;
            fields.push((name, ty));

            if self.check(&Token::Comma) {
                self.advance();
            }
        }
        Ok(fields)
    }

    fn parse_enum(&mut self) -> Result<Item, String> {
        self.advance(); // consume 'enum' or ᛈ

        let name = match self.advance() {
            Token::Ident(s) => s,
            t => return Err(format!("Expected enum name, got {:?}", t)),
        };

        self.expect(Token::LBrace)?;
        let variants = self.parse_enum_variants()?;
        self.expect(Token::RBrace)?;

        Ok(Item::Enum { name, variants })
    }

    fn parse_enum_variants(&mut self) -> Result<Vec<(String, Option<Vec<String>>)>, String> {
        let mut variants = Vec::new();
        while !self.check(&Token::RBrace) {
            let name = match self.advance() {
                Token::Ident(s) => s,
                _ => break,
            };

            let fields = if self.check(&Token::LParen) {
                self.advance();
                let mut f = Vec::new();
                while !self.check(&Token::RParen) {
                    f.push(self.parse_type()?);
                    if self.check(&Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(Token::RParen)?;
                Some(f)
            } else {
                None
            };

            variants.push((name, fields));

            if self.check(&Token::Comma) {
                self.advance();
            }
        }
        Ok(variants)
    }

    fn parse_impl(&mut self) -> Result<Item, String> {
        self.advance(); // consume 'impl' or ⊢

        let name = match self.advance() {
            Token::Ident(s) => s,
            t => return Err(format!("Expected type name, got {:?}", t)),
        };

        self.expect(Token::LBrace)?;
        let mut methods = Vec::new();
        while !self.check(&Token::RBrace) {
            methods.push(self.parse_function()?);
        }
        self.expect(Token::RBrace)?;

        Ok(Item::Impl { name, methods })
    }

    fn parse_block_contents(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = Vec::new();
        while !self.check(&Token::RBrace) {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        match &self.current {
            Token::Assign => {
                self.advance();
                let name = match self.advance() {
                    Token::Ident(s) => s,
                    t => return Err(format!("Expected variable name, got {:?}", t)),
                };

                let ty = if self.check(&Token::Colon) {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };

                self.expect(Token::Eq)?;
                let value = self.parse_expr()?;

                if self.check(&Token::Semi) {
                    self.advance();
                }

                Ok(Stmt::Let { name, ty, value, mutable: false })
            }
            Token::Vary => {
                self.advance();
                let name = match self.advance() {
                    Token::Ident(s) => s,
                    t => return Err(format!("Expected variable name, got {:?}", t)),
                };

                let ty = if self.check(&Token::Colon) {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };

                self.expect(Token::Eq)?;
                let value = self.parse_expr()?;

                if self.check(&Token::Semi) {
                    self.advance();
                }

                Ok(Stmt::Let { name, ty, value, mutable: true })
            }
            Token::Return => {
                self.advance();
                let value = if self.check(&Token::Semi) || self.check(&Token::RBrace) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                if self.check(&Token::Semi) {
                    self.advance();
                }
                Ok(Stmt::Return(value))
            }
            Token::While => {
                self.advance();
                let cond = self.parse_expr()?;
                self.expect(Token::LBrace)?;
                let body_stmts = self.parse_block_contents()?;
                self.expect(Token::RBrace)?;
                Ok(Stmt::While {
                    cond,
                    body: Expr::Block(body_stmts)
                })
            }
            _ => {
                let expr = self.parse_expr()?;
                if self.check(&Token::Semi) {
                    self.advance();
                }
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> Result<Expr, String> {
        let expr = self.parse_or()?;

        // Check for assignment: ident = value
        if self.check(&Token::Eq) {
            if let Expr::Ident(name) = &expr {
                self.advance();
                let value = self.parse_assignment()?;
                // Treat assignment as a let statement expression that returns null
                // For now, just handle it as a binary op that updates the variable
                return Ok(Expr::Binary {
                    op: BinOp::Assign,
                    left: Box::new(Expr::Ident(name.clone())),
                    right: Box::new(value),
                });
            }
        }

        Ok(expr)
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_and()?;
        while self.check(&Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::Binary {
                op: BinOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_equality()?;
        while self.check(&Token::And) {
            self.advance();
            let right = self.parse_equality()?;
            left = Expr::Binary {
                op: BinOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_comparison()?;
        loop {
            let op = match &self.current {
                Token::EqEq => BinOp::Eq,
                Token::NotEq => BinOp::NotEq,
                _ => break,
            };
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_term()?;
        loop {
            let op = match &self.current {
                Token::Lt => BinOp::Lt,
                Token::LtEq => BinOp::LtEq,
                Token::Gt => BinOp::Gt,
                Token::GtEq => BinOp::GtEq,
                _ => break,
            };
            self.advance();
            let right = self.parse_term()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_factor()?;
        loop {
            let op = match &self.current {
                Token::Plus => BinOp::Add,
                Token::PlusPlus => BinOp::Concat,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_factor()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match &self.current {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        match &self.current {
            Token::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
            Token::Not => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            _ => self.parse_call(),
        }
    }

    fn parse_call(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.check(&Token::LParen) {
                self.advance();
                let args = self.parse_args()?;
                self.expect(Token::RParen)?;
                expr = Expr::Call {
                    func: Box::new(expr),
                    args,
                };
            } else if self.check(&Token::Dot) {
                self.advance();
                let field = match self.advance() {
                    Token::Ident(s) => s,
                    t => return Err(format!("Expected field name, got {:?}", t)),
                };
                expr = Expr::FieldAccess {
                    expr: Box::new(expr),
                    field,
                };
            } else if self.check(&Token::LBracket) {
                self.advance();
                let index = self.parse_expr()?;
                self.expect(Token::RBracket)?;
                expr = Expr::Index {
                    expr: Box::new(expr),
                    index: Box::new(index),
                };
            } else if self.check(&Token::Pipe) {
                // Morpheme pipe: expr |τ{...} or expr |Σ
                self.advance(); // consume |
                let (op, closure) = self.parse_morpheme_op()?;
                expr = Expr::Morpheme {
                    op,
                    expr: Box::new(expr),
                    closure,
                };
            } else if self.check(&Token::ColonColon) {
                // Static method call: Type·method() or Type::method()
                self.advance();
                let method = match self.advance() {
                    Token::Ident(s) => s,
                    t => return Err(format!("Expected method name, got {:?}", t)),
                };
                // Check for call
                if self.check(&Token::LParen) {
                    self.advance();
                    let args = self.parse_args()?;
                    self.expect(Token::RParen)?;
                    expr = Expr::Call {
                        func: Box::new(Expr::FieldAccess {
                            expr: Box::new(expr),
                            field: method,
                        }),
                        args,
                    };
                } else {
                    expr = Expr::FieldAccess {
                        expr: Box::new(expr),
                        field: method,
                    };
                }
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_morpheme_op(&mut self) -> Result<(MorphOp, Option<Box<Expr>>), String> {
        let op = match &self.current {
            Token::MorphTau => MorphOp::Tau,
            Token::MorphPhi => MorphOp::Phi,
            Token::MorphSigma => MorphOp::Sigma,
            Token::MorphPi => MorphOp::Pi,
            Token::MorphMu => MorphOp::Mu,
            Token::MorphAlpha => MorphOp::Alpha,
            Token::MorphOmega => MorphOp::Omega,
            Token::MorphLambda => MorphOp::Lambda,
            Token::MorphSort => MorphOp::Sort,
            Token::MorphRho => MorphOp::Rho,
            t => return Err(format!("Expected morpheme operator, got {:?}", t)),
        };
        self.advance();

        // Check for closure: {_ * 2}
        let closure = if self.check(&Token::LBrace) {
            self.advance();
            // Parse closure body - a simple expression using _ as placeholder
            let body = self.parse_expr()?;
            self.expect(Token::RBrace)?;
            Some(Box::new(body))
        } else {
            None
        };

        Ok((op, closure))
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, String> {
        let mut args = Vec::new();
        while !self.check(&Token::RParen) {
            args.push(self.parse_expr()?);
            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.advance() {
            Token::Int(n) => Ok(Expr::Int(n)),
            Token::Float(f) => Ok(Expr::Float(f)),
            Token::True => Ok(Expr::Bool(true)),
            Token::False => Ok(Expr::Bool(false)),
            Token::Str(s) => Ok(Expr::Str(s)),
            Token::Self_ => Ok(Expr::Ident("self".to_string())),
            Token::Underscore => Ok(Expr::Ident("_".to_string())),
            Token::Ident(name) => {
                // Check for struct literal
                if self.check(&Token::LBrace) {
                    self.advance();
                    let fields = self.parse_struct_lit_fields()?;
                    self.expect(Token::RBrace)?;
                    Ok(Expr::StructLit { name, fields })
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            Token::LParen => {
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Token::LBrace => {
                let stmts = self.parse_block_contents()?;
                self.expect(Token::RBrace)?;
                Ok(Expr::Block(stmts))
            }
            Token::LBracket => {
                let mut elements = Vec::new();
                while !self.check(&Token::RBracket) {
                    elements.push(self.parse_expr()?);
                    if !self.check(&Token::Comma) {
                        break;
                    }
                    self.advance();
                }
                self.expect(Token::RBracket)?;
                Ok(Expr::Array(elements))
            }
            Token::If => self.parse_if(),
            Token::Match => self.parse_match(),
            t => Err(format!("Unexpected token: {:?}", t)),
        }
    }

    fn parse_struct_lit_fields(&mut self) -> Result<Vec<(String, Expr)>, String> {
        let mut fields = Vec::new();
        while !self.check(&Token::RBrace) {
            let name = match self.advance() {
                Token::Ident(s) => s,
                _ => break,
            };
            self.expect(Token::Colon)?;
            let value = self.parse_expr()?;
            fields.push((name, value));

            if self.check(&Token::Comma) {
                self.advance();
            }
        }
        Ok(fields)
    }

    fn parse_if(&mut self) -> Result<Expr, String> {
        let cond = self.parse_expr()?;
        self.expect(Token::LBrace)?;
        let then_stmts = self.parse_block_contents()?;
        self.expect(Token::RBrace)?;

        let else_ = if self.check(&Token::Else) {
            self.advance();
            if self.check(&Token::If) {
                self.advance();
                Some(Box::new(self.parse_if()?))
            } else {
                self.expect(Token::LBrace)?;
                let else_stmts = self.parse_block_contents()?;
                self.expect(Token::RBrace)?;
                Some(Box::new(Expr::Block(else_stmts)))
            }
        } else {
            None
        };

        Ok(Expr::If {
            cond: Box::new(cond),
            then_: Box::new(Expr::Block(then_stmts)),
            else_,
        })
    }

    /// Parse the subject of a match expression.
    /// This is a simple expression that doesn't consume struct literals.
    fn parse_match_subject(&mut self) -> Result<Expr, String> {
        // Parse just an identifier or simple expression
        match self.advance() {
            Token::Int(n) => Ok(Expr::Int(n)),
            Token::Float(f) => Ok(Expr::Float(f)),
            Token::True => Ok(Expr::Bool(true)),
            Token::False => Ok(Expr::Bool(false)),
            Token::Str(s) => Ok(Expr::Str(s)),
            Token::Self_ => Ok(Expr::Ident("self".to_string())),
            Token::Ident(name) => {
                // Just return the identifier, don't look for struct literal
                let mut expr = Expr::Ident(name);
                // Allow field access
                while self.check(&Token::Dot) {
                    self.advance();
                    let field = match self.advance() {
                        Token::Ident(s) => s,
                        t => return Err(format!("Expected field name, got {:?}", t)),
                    };
                    expr = Expr::FieldAccess {
                        expr: Box::new(expr),
                        field,
                    };
                }
                Ok(expr)
            }
            Token::LParen => {
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            t => Err(format!("Unexpected token in match subject: {:?}", t)),
        }
    }

    fn parse_match(&mut self) -> Result<Expr, String> {
        // Parse a simple expression (not a full expression to avoid struct literal confusion)
        let expr = self.parse_match_subject()?;
        self.expect(Token::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(&Token::RBrace) {
            let pattern = self.parse_pattern()?;
            self.expect(Token::FatArrow)?;
            let body = self.parse_expr()?;
            arms.push(MatchArm { pattern, body });

            if self.check(&Token::Comma) {
                self.advance();
            }
        }

        self.expect(Token::RBrace)?;
        Ok(Expr::Match { expr: Box::new(expr), arms })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        match self.advance() {
            Token::Underscore => Ok(Pattern::Wildcard),
            Token::Int(n) => Ok(Pattern::Int(n)),
            Token::Ident(name) => {
                if self.check(&Token::LParen) {
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.check(&Token::RParen) {
                        if let Token::Ident(f) = self.advance() {
                            fields.push(f);
                        }
                        if self.check(&Token::Comma) {
                            self.advance();
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Pattern::Variant { name, fields: Some(fields) })
                } else {
                    Ok(Pattern::Ident(name))
                }
            }
            t => Err(format!("Invalid pattern: {:?}", t)),
        }
    }
}
