//! Sigil Lexer - Tokenizes source code
//!
//! Supports canonical Sigil syntax with Unicode operators.

use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Rite,       // rite (function)
    Sigil,      // sigil (struct)
    Enum,       // enum or ᛈ (perthro rune)
    Impl,       // impl or ⊢ (turnstile)
    If,         // if or ⎇
    Else,       // else or ⎉
    Match,      // match or ⌥
    While,
    For,
    In,
    Return,
    True,       // true or yea
    False,      // false or nay
    Self_,
    Vary,       // vary (mutable binding)

    // Identifiers and literals
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),

    // Operators
    Plus,
    PlusPlus,   // ++ string concatenation
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    EqEq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,        // && or ∧
    Or,         // || or ∨
    Not,        // ! or ¬
    Assign,     // ≔ or let binding
    Arrow,      // → or ->
    FatArrow,   // =>
    Dot,
    Colon,
    ColonColon, // :: or · (middledot)
    Comma,
    Semi,
    Underscore,
    Pipe,       // | for morphemes

    // Morpheme operators
    MorphTau,   // τ - transform/map
    MorphPhi,   // φ - filter
    MorphSigma, // Σ - sum
    MorphPi,    // Π - product
    MorphMu,    // μ - mean
    MorphAlpha, // α - first
    MorphOmega, // ω - last
    MorphLambda,// λ - length
    MorphSort,  // σ - sort
    MorphRho,   // ρ - reduce

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,

    // Special
    Eof,
}

pub struct Lexer<'a> {
    input: Peekable<Chars<'a>>,
    current: char,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        let mut input = source.chars().peekable();
        let current = input.next().unwrap_or('\0');
        Self {
            input,
            current,
            pos: 0,
        }
    }

    fn advance(&mut self) -> char {
        let c = self.current;
        self.current = self.input.next().unwrap_or('\0');
        self.pos += 1;
        c
    }

    fn peek(&mut self) -> char {
        *self.input.peek().unwrap_or(&'\0')
    }

    fn skip_whitespace(&mut self) {
        while self.current.is_whitespace() {
            self.advance();
        }
    }

    fn skip_comment(&mut self) {
        if self.current == '/' && self.peek() == '/' {
            while self.current != '\n' && self.current != '\0' {
                self.advance();
            }
        }
    }

    fn read_ident(&mut self) -> String {
        let mut s = String::new();
        while self.current.is_alphanumeric() || self.current == '_' {
            s.push(self.advance());
        }
        s
    }

    fn read_number(&mut self) -> Token {
        let mut s = String::new();
        let mut is_float = false;

        while self.current.is_ascii_digit() || self.current == '.' || self.current == '_' {
            if self.current == '.' {
                if is_float {
                    break;
                }
                is_float = true;
            }
            if self.current != '_' {
                s.push(self.current);
            }
            self.advance();
        }

        if is_float {
            Token::Float(s.parse().unwrap_or(0.0))
        } else {
            Token::Int(s.parse().unwrap_or(0))
        }
    }

    fn read_string(&mut self) -> String {
        self.advance(); // skip opening quote
        let mut s = String::new();
        while self.current != '"' && self.current != '\0' {
            if self.current == '\\' {
                self.advance();
                match self.current {
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    '\\' => s.push('\\'),
                    '"' => s.push('"'),
                    _ => s.push(self.current),
                }
            } else {
                s.push(self.current);
            }
            self.advance();
        }
        self.advance(); // skip closing quote
        s
    }

    pub fn next_token(&mut self) -> Token {
        loop {
            self.skip_whitespace();
            self.skip_comment();
            self.skip_whitespace();

            if self.current != '/' || self.peek() != '/' {
                break;
            }
        }

        let token = match self.current {
            '\0' => Token::Eof,

            // Unicode operators - Sigil native syntax
            '→' => { self.advance(); Token::Arrow }
            '≔' => { self.advance(); Token::Assign }
            '⊢' => { self.advance(); Token::Impl }
            'ᛈ' => { self.advance(); Token::Enum }
            '⎇' => { self.advance(); Token::If }      // Sigil if
            '⎉' => { self.advance(); Token::Else }    // Sigil else
            '⌥' => { self.advance(); Token::Match }   // Sigil match
            '∧' => { self.advance(); Token::And }     // Sigil and
            '∨' => { self.advance(); Token::Or }      // Sigil or
            '¬' => { self.advance(); Token::Not }     // Sigil not
            '·' => { self.advance(); Token::ColonColon } // Middledot for method calls

            // Morpheme operators
            'τ' => { self.advance(); Token::MorphTau }
            'φ' => { self.advance(); Token::MorphPhi }
            'Σ' => { self.advance(); Token::MorphSigma }
            'Π' => { self.advance(); Token::MorphPi }
            'μ' => { self.advance(); Token::MorphMu }
            'α' => { self.advance(); Token::MorphAlpha }
            'ω' => { self.advance(); Token::MorphOmega }
            'λ' => { self.advance(); Token::MorphLambda }
            'σ' => { self.advance(); Token::MorphSort }
            'ρ' => { self.advance(); Token::MorphRho }

            // ASCII operators
            '+' => {
                self.advance();
                if self.current == '+' {
                    self.advance();
                    Token::PlusPlus
                } else {
                    Token::Plus
                }
            }
            '*' => { self.advance(); Token::Star }
            '/' => { self.advance(); Token::Slash }
            '%' => { self.advance(); Token::Percent }
            '.' => { self.advance(); Token::Dot }
            ',' => { self.advance(); Token::Comma }
            ';' => { self.advance(); Token::Semi }
            '(' => { self.advance(); Token::LParen }
            ')' => { self.advance(); Token::RParen }
            '{' => { self.advance(); Token::LBrace }
            '}' => { self.advance(); Token::RBrace }
            '[' => { self.advance(); Token::LBracket }
            ']' => { self.advance(); Token::RBracket }

            '-' => {
                self.advance();
                if self.current == '>' {
                    self.advance();
                    Token::Arrow
                } else {
                    Token::Minus
                }
            }

            '=' => {
                self.advance();
                if self.current == '=' {
                    self.advance();
                    Token::EqEq
                } else if self.current == '>' {
                    self.advance();
                    Token::FatArrow
                } else {
                    Token::Eq
                }
            }

            '!' => {
                self.advance();
                if self.current == '=' {
                    self.advance();
                    Token::NotEq
                } else {
                    Token::Not
                }
            }

            '<' => {
                self.advance();
                if self.current == '=' {
                    self.advance();
                    Token::LtEq
                } else {
                    Token::Lt
                }
            }

            '>' => {
                self.advance();
                if self.current == '=' {
                    self.advance();
                    Token::GtEq
                } else {
                    Token::Gt
                }
            }

            '&' => {
                self.advance();
                if self.current == '&' {
                    self.advance();
                }
                Token::And
            }

            '|' => {
                self.advance();
                if self.current == '|' {
                    self.advance();
                    Token::Or
                } else {
                    Token::Pipe
                }
            }

            ':' => {
                self.advance();
                if self.current == ':' {
                    self.advance();
                    Token::ColonColon
                } else {
                    Token::Colon
                }
            }

            '_' => {
                if self.peek().is_alphanumeric() {
                    Token::Ident(self.read_ident())
                } else {
                    self.advance();
                    Token::Underscore
                }
            }

            '"' => Token::Str(self.read_string()),

            c if c.is_ascii_digit() => self.read_number(),

            c if c.is_alphabetic() || c == '_' => {
                let ident = self.read_ident();
                match ident.as_str() {
                    "rite" | "fn" => Token::Rite,
                    "sigil" | "struct" => Token::Sigil,
                    "enum" => Token::Enum,
                    "impl" => Token::Impl,
                    "if" => Token::If,
                    "else" => Token::Else,
                    "match" => Token::Match,
                    "while" => Token::While,
                    "for" => Token::For,
                    "in" => Token::In,
                    "return" => Token::Return,
                    "true" | "yea" => Token::True,
                    "false" | "nay" => Token::False,
                    "self" => Token::Self_,
                    "let" => Token::Assign, // let is also binding
                    "vary" => Token::Vary,  // mutable binding
                    _ => Token::Ident(ident),
                }
            }

            _ => {
                self.advance();
                return self.next_token();
            }
        };

        token
    }
}
