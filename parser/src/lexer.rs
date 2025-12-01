//! Lexer for the Sigil programming language.
//!
//! Handles polysynthetic morphemes, evidentiality markers, and multi-base numerals.

use logos::Logos;
use crate::span::Span;

/// Token types for Sigil.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n\f]+")]
pub enum Token {
    // === Comments ===
    #[regex(r"//[^\n]*", |lex| lex.slice().to_string())]
    LineComment(String),

    #[regex(r"//![^\n]*", |lex| lex.slice().to_string())]
    DocComment(String),

    // === Keywords ===
    #[token("fn")]
    Fn,
    #[token("async")]
    Async,
    #[token("let")]
    Let,
    #[token("mut")]
    Mut,
    #[token("const")]
    Const,
    #[token("type")]
    Type,
    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("trait")]
    Trait,
    #[token("impl")]
    Impl,
    #[token("mod")]
    Mod,
    #[token("use")]
    Use,
    #[token("pub")]
    Pub,
    #[token("actor")]
    Actor,
    #[token("saga")]
    Saga,
    #[token("scope")]
    Scope,
    #[token("rune")]
    Rune,

    // Control flow
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("loop")]
    Loop,
    #[token("while")]
    While,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,
    #[token("yield")]
    Yield,
    #[token("await")]
    Await,

    // Other keywords
    #[token("self")]
    SelfLower,
    #[token("Self")]
    SelfUpper,
    #[token("super")]
    Super,
    #[token("crate")]
    Crate,
    #[token("where")]
    Where,
    #[token("as")]
    As,
    #[token("dyn")]
    Dyn,
    #[token("move")]
    Move,
    #[token("ref")]
    Ref,
    #[token("static")]
    Static,
    #[token("unsafe")]
    Unsafe,
    #[token("extern")]
    Extern,
    #[token("on")]
    On,

    // Boolean literals
    #[token("true")]
    True,
    #[token("false")]
    False,

    // === Morphemes (Greek letters) ===
    #[token("τ")]
    #[token("Τ")]
    Tau,  // Transform/map

    #[token("φ")]
    #[token("Φ")]
    Phi,  // Filter

    #[token("σ")]
    #[token("Σ")]
    Sigma,  // Sort (lowercase) / Sum (uppercase)

    #[token("ρ")]
    #[token("Ρ")]
    Rho,  // Reduce

    #[token("λ")]
    #[token("Λ")]
    Lambda,  // Lambda

    #[token("Π")]
    Pi,  // Product

    #[token("⌛")]
    Hourglass,  // Await symbol

    // Additional morphemes
    #[token("δ")]
    #[token("Δ")]
    Delta,  // Difference/change

    #[token("ε")]
    Epsilon,  // Empty/null

    #[token("ω")]
    #[token("Ω")]
    Omega,  // End/terminal

    #[token("α")]
    Alpha,  // First element

    #[token("ζ")]
    Zeta,  // Zip/combine

    // === Quantifiers (for AI-native set operations) ===
    #[token("∀")]
    ForAll,  // Universal quantification

    #[token("∃")]
    Exists,  // Existential quantification

    #[token("∈")]
    ElementOf,  // Membership test

    #[token("∉")]
    NotElementOf,  // Non-membership

    // === Set Operations ===
    #[token("∪")]
    Union,  // Set union

    #[token("∩")]
    Intersection,  // Set intersection

    #[token("∖")]
    SetMinus,  // Set difference

    #[token("⊂")]
    Subset,  // Proper subset

    #[token("⊆")]
    SubsetEq,  // Subset or equal

    #[token("⊃")]
    Superset,  // Proper superset

    #[token("⊇")]
    SupersetEq,  // Superset or equal

    // === Logic Operators ===
    #[token("∧")]
    LogicAnd,  // Logical conjunction

    #[token("∨")]
    LogicOr,  // Logical disjunction

    #[token("¬")]
    LogicNot,  // Logical negation

    #[token("⊻")]
    LogicXor,  // Exclusive or

    #[token("⊤")]
    Top,  // True/any type

    #[token("⊥")]
    Bottom,  // False/never type

    // === Type Theory ===
    #[token("∷")]
    TypeAnnotation,  // Type annotation (alternative to :)

    // === Analysis/Calculus ===
    #[token("∫")]
    Integral,  // Cumulative sum

    #[token("∂")]
    Partial,  // Discrete derivative

    #[token("√")]
    Sqrt,  // Square root

    #[token("∛")]
    Cbrt,  // Cube root

    // === Category Theory ===
    #[token("∘")]
    Compose,  // Function composition

    #[token("⊗")]
    Tensor,  // Tensor product

    #[token("⊕")]
    DirectSum,  // Direct sum / XOR

    // === Evidentiality Markers ===
    // Note: These are handled contextually since ! and ? have other uses
    #[token("‽")]
    Interrobang,  // Paradox/trust boundary

    // === Operators ===
    #[token("|")]
    Pipe,
    #[token("·")]
    MiddleDot,  // Incorporation
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("<-")]
    LeftArrow,
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("**")]
    StarStar,  // Exponentiation
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("!")]
    Bang,  // Evidentiality: known / logical not
    #[token("?")]
    Question,  // Evidentiality: uncertain / try
    #[token("~")]
    Tilde,  // Evidentiality: reported
    #[token("&")]
    Amp,
    #[token("^")]
    Caret,
    #[token("<<")]
    Shl,
    #[token(">>")]
    Shr,
    #[token("=")]
    Eq,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("..")]
    DotDot,
    #[token("..=")]
    DotDotEq,
    #[token("++")]
    PlusPlus,  // Concatenation
    #[token("::")]
    ColonColon,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("@")]
    At,
    #[token("#")]
    Hash,
    #[token("_", priority = 3)]
    Underscore,

    // === Delimiters ===
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,

    // === Special symbols ===
    #[token("∅")]
    Empty,  // Void/emptiness (śūnya)
    #[token("◯")]
    Circle,  // Geometric zero
    #[token("∞")]
    Infinity,  // Ananta

    // === Numbers ===
    // Binary: 0b...
    #[regex(r"0b[01_]+", |lex| lex.slice().to_string())]
    BinaryLit(String),

    // Octal: 0o...
    #[regex(r"0o[0-7_]+", |lex| lex.slice().to_string())]
    OctalLit(String),

    // Hex: 0x...
    #[regex(r"0x[0-9a-fA-F_]+", |lex| lex.slice().to_string())]
    HexLit(String),

    // Vigesimal: 0v... (base 20)
    #[regex(r"0v[0-9a-jA-J_]+", |lex| lex.slice().to_string())]
    VigesimalLit(String),

    // Sexagesimal: 0s... (base 60)
    #[regex(r"0s[0-9a-zA-Z_]+", |lex| lex.slice().to_string())]
    SexagesimalLit(String),

    // Duodecimal: 0z... (base 12)
    #[regex(r"0z[0-9a-bA-B_]+", |lex| lex.slice().to_string())]
    DuodecimalLit(String),

    // Float: 123.456 or 1.23e10
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9_]+)?", |lex| lex.slice().to_string())]
    FloatLit(String),

    // Integer: 123
    #[regex(r"[0-9][0-9_]*", |lex| lex.slice().to_string())]
    IntLit(String),

    // === Strings ===
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        s[1..s.len()-1].to_string()
    })]
    StringLit(String),

    // Char literal
    #[regex(r"'([^'\\]|\\.)'", |lex| {
        let s = lex.slice();
        s.chars().nth(1).unwrap()
    })]
    CharLit(char),

    // Raw string (simplified - just r"..." for now)
    #[regex(r#"r"[^"]*""#, |lex| lex.slice().to_string())]
    RawStringLit(String),

    // === Identifiers ===
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // === Rune annotation ===
    #[regex(r"//@\s*rune:\s*[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    RuneAnnotation(String),
}

impl Token {
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            Token::Fn | Token::Async | Token::Let | Token::Mut | Token::Const |
            Token::Type | Token::Struct | Token::Enum | Token::Trait | Token::Impl |
            Token::Mod | Token::Use | Token::Pub | Token::Actor | Token::Saga |
            Token::Scope | Token::Rune | Token::If | Token::Else | Token::Match |
            Token::Loop | Token::While | Token::For | Token::In | Token::Break |
            Token::Continue | Token::Return | Token::Yield | Token::Await
        )
    }

    pub fn is_morpheme(&self) -> bool {
        matches!(
            self,
            Token::Tau | Token::Phi | Token::Sigma | Token::Rho |
            Token::Lambda | Token::Pi | Token::Hourglass |
            Token::Delta | Token::Epsilon | Token::Omega | Token::Alpha | Token::Zeta |
            Token::Integral | Token::Partial | Token::Sqrt | Token::Cbrt |
            Token::Compose
        )
    }

    pub fn is_quantifier(&self) -> bool {
        matches!(
            self,
            Token::ForAll | Token::Exists | Token::ElementOf | Token::NotElementOf
        )
    }

    pub fn is_set_op(&self) -> bool {
        matches!(
            self,
            Token::Union | Token::Intersection | Token::SetMinus |
            Token::Subset | Token::SubsetEq | Token::Superset | Token::SupersetEq
        )
    }

    pub fn is_logic_op(&self) -> bool {
        matches!(
            self,
            Token::LogicAnd | Token::LogicOr | Token::LogicNot | Token::LogicXor |
            Token::Top | Token::Bottom
        )
    }

    pub fn is_evidentiality(&self) -> bool {
        matches!(
            self,
            Token::Bang | Token::Question | Token::Tilde | Token::Interrobang
        )
    }
}

/// Lexer wrapping Logos for Sigil.
pub struct Lexer<'a> {
    inner: logos::Lexer<'a, Token>,
    peeked: Option<Option<(Token, Span)>>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            inner: Token::lexer(source),
            peeked: None,
        }
    }

    pub fn next_token(&mut self) -> Option<(Token, Span)> {
        if let Some(peeked) = self.peeked.take() {
            return peeked;
        }

        match self.inner.next() {
            Some(Ok(token)) => {
                let span = self.inner.span();
                Some((token, Span::new(span.start, span.end)))
            }
            Some(Err(_)) => {
                // Skip invalid tokens and try next
                self.next_token()
            }
            None => None,
        }
    }

    pub fn peek(&mut self) -> Option<&(Token, Span)> {
        if self.peeked.is_none() {
            self.peeked = Some(self.next_token());
        }
        self.peeked.as_ref().and_then(|p| p.as_ref())
    }

    pub fn span(&self) -> Span {
        let span = self.inner.span();
        Span::new(span.start, span.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphemes() {
        let mut lexer = Lexer::new("τ φ σ ρ λ Σ Π ⌛");
        assert!(matches!(lexer.next_token(), Some((Token::Tau, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Phi, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Sigma, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Rho, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Lambda, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Sigma, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Pi, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Hourglass, _))));
    }

    #[test]
    fn test_evidentiality() {
        let mut lexer = Lexer::new("value! uncertain? reported~ paradox‽");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "value"));
        assert!(matches!(lexer.next_token(), Some((Token::Bang, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "uncertain"));
        assert!(matches!(lexer.next_token(), Some((Token::Question, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "reported"));
        assert!(matches!(lexer.next_token(), Some((Token::Tilde, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "paradox"));
        assert!(matches!(lexer.next_token(), Some((Token::Interrobang, _))));
    }

    #[test]
    fn test_pipe_chain() {
        let mut lexer = Lexer::new("data|τ{f}|φ{p}|σ");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "data"));
        assert!(matches!(lexer.next_token(), Some((Token::Pipe, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Tau, _))));
        assert!(matches!(lexer.next_token(), Some((Token::LBrace, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "f"));
        assert!(matches!(lexer.next_token(), Some((Token::RBrace, _))));
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("42 0b1010 0o52 0x2A 0v22 0s42 3.14");
        assert!(matches!(lexer.next_token(), Some((Token::IntLit(s), _)) if s == "42"));
        assert!(matches!(lexer.next_token(), Some((Token::BinaryLit(s), _)) if s == "0b1010"));
        assert!(matches!(lexer.next_token(), Some((Token::OctalLit(s), _)) if s == "0o52"));
        assert!(matches!(lexer.next_token(), Some((Token::HexLit(s), _)) if s == "0x2A"));
        assert!(matches!(lexer.next_token(), Some((Token::VigesimalLit(s), _)) if s == "0v22"));
        assert!(matches!(lexer.next_token(), Some((Token::SexagesimalLit(s), _)) if s == "0s42"));
        assert!(matches!(lexer.next_token(), Some((Token::FloatLit(s), _)) if s == "3.14"));
    }

    #[test]
    fn test_incorporation() {
        let mut lexer = Lexer::new("file·open·read");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "file"));
        assert!(matches!(lexer.next_token(), Some((Token::MiddleDot, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "open"));
        assert!(matches!(lexer.next_token(), Some((Token::MiddleDot, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "read"));
    }

    #[test]
    fn test_special_symbols() {
        let mut lexer = Lexer::new("∅ ◯ ∞");
        assert!(matches!(lexer.next_token(), Some((Token::Empty, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Circle, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Infinity, _))));
    }

    #[test]
    fn test_quantifiers() {
        let mut lexer = Lexer::new("∀x ∃y x∈S y∉T");
        assert!(matches!(lexer.next_token(), Some((Token::ForAll, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "x"));
        assert!(matches!(lexer.next_token(), Some((Token::Exists, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "y"));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "x"));
        assert!(matches!(lexer.next_token(), Some((Token::ElementOf, _))));
    }

    #[test]
    fn test_set_operations() {
        let mut lexer = Lexer::new("A∪B A∩B A∖B A⊂B A⊆B");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "A"));
        assert!(matches!(lexer.next_token(), Some((Token::Union, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "B"));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "A"));
        assert!(matches!(lexer.next_token(), Some((Token::Intersection, _))));
    }

    #[test]
    fn test_logic_operators() {
        let mut lexer = Lexer::new("p∧q p∨q ¬p p⊻q ⊤ ⊥");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "p"));
        assert!(matches!(lexer.next_token(), Some((Token::LogicAnd, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "q"));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "p"));
        assert!(matches!(lexer.next_token(), Some((Token::LogicOr, _))));
    }

    #[test]
    fn test_analysis_operators() {
        let mut lexer = Lexer::new("∫f ∂g √x ∛y f∘g");
        assert!(matches!(lexer.next_token(), Some((Token::Integral, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "f"));
        assert!(matches!(lexer.next_token(), Some((Token::Partial, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "g"));
        assert!(matches!(lexer.next_token(), Some((Token::Sqrt, _))));
    }

    #[test]
    fn test_additional_morphemes() {
        let mut lexer = Lexer::new("δ ε ω α ζ");
        assert!(matches!(lexer.next_token(), Some((Token::Delta, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Epsilon, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Omega, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Alpha, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Zeta, _))));
    }

    #[test]
    fn test_ffi_keywords() {
        let mut lexer = Lexer::new("extern unsafe");
        assert!(matches!(lexer.next_token(), Some((Token::Extern, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Unsafe, _))));
    }
}
